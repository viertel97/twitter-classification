import json
import os

from dotenv import load_dotenv
from openai.types.fine_tuning import FineTuningJob
from quarter_lib.logging import setup_logging

from src.config import SEED
from src.config.llm_config import get_function_schema, get_messages

logger = setup_logging(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")


if OPENAI_API_KEY:
	from openai import OpenAI

	client = OpenAI(api_key=OPENAI_API_KEY)
elif AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
	from openai import AzureOpenAI

	client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, azure_endpoint=AZURE_OPENAI_ENDPOINT, api_version="2024-08-01-preview")
else:
	raise Exception("No OpenAI API key found.")


def classify_conversation(conversation_data: dict[str, list[str]], companies: list[str], model: str) -> str:
	allowed_companies = companies.copy()
	if "Unclear" not in allowed_companies:
		allowed_companies.append("Unclear")

	function_schema = get_function_schema(allowed_companies)
	messages = get_messages(conversation_data)

	response = client.beta.chat.completions.parse(
		model=model.replace(".", "-"), seed=SEED, messages=messages, functions=[function_schema], function_call={"name": "classify_company"}
	)

	message = response.choices[0].message

	if message.function_call:
		arguments = json.loads(message.function_call.arguments)
	else:
		content = message.content
		arguments = json.loads(content)
	try:
		if "company" not in arguments.keys():
			logger.error("No 'company' key in function call arguments.")
			return "Unclear"
		return arguments["company"]
	except (json.JSONDecodeError, Exception) as e:
		logger.error(f"Error parsing function call arguments: {e}")
		return "Unclear"


def upload_training_and_validation_files(training_file_path: str, validation_file_path: str) -> tuple[str, str]:
	training_response = client.files.create(file=open(training_file_path, "rb"), purpose="fine-tune")
	training_file_id = training_response.id

	validation_response = client.files.create(file=open(validation_file_path, "rb"), purpose="fine-tune")
	validation_file_id = validation_response.id

	logger.info(f"Training file ID: {training_file_id}")
	logger.info(f"Validation file ID: {validation_file_id}")

	return training_file_id, validation_file_id


def start_fine_tuning_job(training_file_id: str, validation_file_id: str, model: str, hyperparameters: dict, seed: int) -> FineTuningJob:
	hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}

	response = client.fine_tuning.jobs.create(
		training_file=training_file_id,
		validation_file=validation_file_id,
		model=model,
		seed=seed,
		hyperparameters=hyperparameters,
	)

	logger.info(f"Fine-tuning job submitted: {response}")
	return response


def check_status(job_id: str) -> tuple[str, str]:
	response = client.fine_tuning.jobs.retrieve(job_id)
	logger.info(f"Job status: {response.status}")
	logger.info(f"Job ID: {response.id}")
	logger.info(f"Model: {response.model_dump_json(indent=2)}")
	status = response.status
	logger.info(f"Status: {status}")
	return_string = f"Job ID: {response.id}\nStatus: {response.status}\n{response.model_dump_json(indent=2)}"
	fine_tuned_model = response.fine_tuned_model
	return return_string, fine_tuned_model

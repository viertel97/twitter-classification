import json
import os

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel
from quarter_lib.logging import setup_logging

from src.config import SEED

logger = setup_logging(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
if not AZURE_ENDPOINT:
    AZURE_ENDPOINT = os.getenv("AZURE_CUSTOM_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")

client = AzureOpenAI(api_key=OPENAI_API_KEY, azure_endpoint=AZURE_ENDPOINT, api_version="2024-08-01-preview")


class ClassificationResult(BaseModel):
    company: str

def get_function_schema(allowed_companies):
    function_schema = {
        "name": "classify_company",
        "description": "Classify the company referred to in the tweet conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "enum": allowed_companies,
                    "description": "The company name that best fits the conversation, or 'Unclear' if none applies."
                }
            },
            "required": ["company"]
        }
    }
    return function_schema


def get_messages(conversation_data):
    return [
        {
            "role": "user",
            "content": f"Please classify the following tweet conversation to one of the allowed companies: {json.dumps(conversation_data)}"
        }
    ]

def classify_conversation(conversation_data, model, companies):
    allowed_companies = companies.copy()
    if "Unclear" not in allowed_companies:
        allowed_companies.append("Unclear")

    function_schema = get_function_schema(allowed_companies)
    messages = get_messages(conversation_data)


    response = client.beta.chat.completions.parse(
        model=model.replace(".", "-"),
        seed=SEED,
        messages=messages,
        functions=[function_schema],
        function_call={"name": "classify_company"}
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


def upload_training_and_validation_files(training_file_path, validation_file_path):
    training_response = client.files.create(
        file=open(training_file_path, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response.id

    validation_response = client.files.create(
        file=open(validation_file_path, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response.id

    logger.info(f"Training file ID: {training_file_id}")
    logger.info(f"Validation file ID: {validation_file_id}")

    return training_file_id, validation_file_id

def start_fine_tuning_job(training_file_id, validation_file_id, model, seed):
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        seed=seed,
        hyperparameters={"n_epochs": 3, "batch_size": 1, "learning_rate_multiplier": 1} # defaults: n_epochs=3, batch_size=1, learning_rate_multiplier=1
    )

    logger.info(f"Fine-tuning job submitted: {response}")
    return response.model_dump_json()

def check_status(job_id):
    response = client.fine_tuning.jobs.retrieve(job_id)
    logger.info(f"Job status: {response.status}")
    logger.info(f"Job ID: {response.id}")
    logger.info(f"Model: {response.model_dump_json(indent=2)}")
    status = response.status
    logger.info(f'Status: {status}')
    return_string = f"Job ID: {response.id}\nStatus: {response.status}\n{response.model_dump_json(indent=2)}"
    fine_tuned_model = response.fine_tuned_model
    return return_string, fine_tuned_model
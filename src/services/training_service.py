import os
from datetime import datetime

from dotenv import load_dotenv
from openai import AzureOpenAI
from quarter_lib.logging import setup_logging

logger = setup_logging(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

client = AzureOpenAI(api_key=OPENAI_API_KEY, azure_endpoint=AZURE_ENDPOINT, api_version="2024-08-01-preview")

global LAST_START

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
    global LAST_START
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        seed=seed
    )

    logger.info(f"Fine-tuning job submitted: {response}")
    LAST_START = datetime.now()
    return response.id

def check_status(job_id):
    global LAST_START
    response = client.fine_tuning.jobs.retrieve(job_id)
    logger.info(f"Job status: {response.status}")
    logger.info(f"Job ID: {response.id}")
    logger.info(f"Model: {response.model_dump_json(indent=2)}")
    elapsed_time = datetime.now() - LAST_START
    status = response.status
    logger.info(f'Status: {status}')
    return_string = f"Job ID: {response.id}\nStatus: {response.status}\n{response.model_dump_json(indent=2)}\nElapsed time: {elapsed_time}"
    return return_string
from typing import Annotated

from fastapi import APIRouter
from fastapi import File, UploadFile
from quarter_lib.logging import setup_logging
from sklearn.model_selection import train_test_split

from src.config import SEED, training_file_path, \
    validation_file_path
from src.services.data_service import prepare_data, create_hierarchical_data, save_to_jsonl
from src.services.openai_service import check_status, upload_training_and_validation_files, start_fine_tuning_job

logger = setup_logging(__name__)

router = APIRouter()



@router.post("/train")
async def train(training_data: Annotated[UploadFile, File], model: str = "gpt-4o-mini-2024-07-18"):
    try:
        contents = training_data.file.read()
        with open(training_data.filename, 'wb') as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {"status": "error"}
    finally:
        training_data.file.close()
    df = prepare_data(training_data.filename)
    conversation_data, companies = create_hierarchical_data(df)

    train_datatest, validation_dataset = train_test_split(conversation_data, test_size=0.2, random_state=SEED)

    save_to_jsonl(train_datatest, training_file_path)
    save_to_jsonl(validation_dataset, validation_file_path)

    training_file_id, validation_file_id = upload_training_and_validation_files(training_file_path,validation_file_path)

    job_response = start_fine_tuning_job(training_file_id, validation_file_id, model, SEED)

    return {"job": job_response, "companies": list(companies), "training_file_id": training_file_id, "validation_file_id": validation_file_id}




@router.get("/status/{job_id}")
async def get_status(job_id: str):
    status, fine_tuned_model = check_status(job_id)

    return {"status": status, "fine_tuned_model": fine_tuned_model}


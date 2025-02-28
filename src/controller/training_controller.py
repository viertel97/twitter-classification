from fastapi import APIRouter
from sklearn.model_selection import train_test_split

from src.services.data_service import prepare_data, create_hierarchical_data, save_to_jsonl
from src.services.training_service import upload_training_and_validation_files, start_fine_tuning_job

router = APIRouter()

from fastapi import File, UploadFile
from quarter_lib.logging import setup_logging
from src.config import SEED, training_file_path, \
    validation_file_path

logger = setup_logging(__name__)




@router.post("/upload")
async def upload_file(training_data: UploadFile = File(...)):
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
    conversation_data = create_hierarchical_data(df)

    train_datatest, validation_dataset = train_test_split(conversation_data, test_size=0.2, random_state=SEED)

    save_to_jsonl(train_datatest, training_file_path)
    save_to_jsonl(validation_dataset, validation_file_path)

    training_file_id, validation_file_id = upload_training_and_validation_files(training_file_path, validation_file_path)

    job_id = start_fine_tuning_job(training_file_id, validation_file_id, "gpt-4o-mini-2024-07-18", SEED)

    return {"status": "ok", "job_id": job_id}





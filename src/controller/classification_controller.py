import pandas as pd
from fastapi import APIRouter
from sklearn.model_selection import train_test_split
from starlette.responses import FileResponse

from src.services.classification_service import classify_conversation
from src.services.data_service import prepare_data, create_hierarchical_data, save_to_jsonl, \
    create_hierarchical_data_prod
from src.services.training_service import upload_training_and_validation_files, start_fine_tuning_job

router = APIRouter()

from fastapi import File, UploadFile
from quarter_lib.logging import setup_logging

logger = setup_logging(__name__)


@router.post("/classify")
async def upload_file(classification_file: UploadFile = File(...), allowed_companies: list = []):
    try:
        contents = classification_file.file.read()
        with open(classification_file.filename, 'wb') as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {"status": "error"}
    finally:
        classification_file.file.close()
    df = prepare_data_prod(classification_file.filename)
    conversation_data = create_hierarchical_data_prod(df)

    results = []
    for conversation in conversation_data:
        classification_result = classify_conversation(conversation, model="gpt-3.5-turbo", companies=allowed_companies)
        results.append({"conversation": conversation, "result": classification_result.company})

    df = pd.DataFrame(results)
    df.to_csv("classification_results.csv")
    return FileResponse("classification_results.csv")




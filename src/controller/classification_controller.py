from typing import Annotated
from tqdm import tqdm

import pandas as pd
from fastapi import APIRouter, Form, File, UploadFile
from quarter_lib.logging import setup_logging
from starlette.responses import FileResponse

from src.config import DEFAULT_COMPANIES
from src.services.data_service import prepare_data, create_hierarchical_data_prod
from src.services.openai_service import classify_conversation

logger = setup_logging(__name__)
router = APIRouter()


@router.post("/classify")
async def upload_file(classification_file: Annotated[UploadFile, File()], allowed_companies: Annotated[str, Form()] = None, fine_tuned_model: Annotated[str, Form()] = None):
    try:
        contents = classification_file.file.read()
        with open(classification_file.filename, 'wb') as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {"status": "error"}
    finally:
        classification_file.file.close()
    if not allowed_companies:
        allowed_companies = DEFAULT_COMPANIES
    df = prepare_data(classification_file.filename)
    conversation_data = create_hierarchical_data_prod(df)

    for conversation in tqdm(conversation_data):
        classification_result = classify_conversation(conversation, model=fine_tuned_model, companies=allowed_companies)
        conversation.update({"company": classification_result})

    df_conversation = pd.DataFrame(conversation_data)
    df_conversation = df_conversation.explode(['conversations', 'tweet_id'])
    df = df.merge(df_conversation[["tweet_id","company"]], on='tweet_id')
    df.to_csv("classification_results.csv", index=False)
    return FileResponse("classification_results.csv")




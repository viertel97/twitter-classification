from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Form, File, UploadFile
from quarter_lib.logging import setup_logging
from starlette.responses import FileResponse
from tqdm import tqdm

from src.config import DEFAULT_COMPANY_BINS, DEFAULT_COMPANY_BINS_WITHOUT_COMPANIES
from src.services.data_service import prepare_data, create_hierarchical_data_prod
from src.services.openai_service import classify_conversation

logger = setup_logging(__name__)
router = APIRouter(tags=["classification"])


@router.post("/classify")
async def upload_file(
	classification_file: Annotated[UploadFile, File()],
	allowed_company_bins: Annotated[dict, Form()] = None,
	fine_tuned_model: Annotated[str, Form()] = None,
):
	try:
		contents = classification_file.file.read()
		with open(classification_file.filename, "wb") as f:
			f.write(contents)
	except Exception as e:
		logger.error(f"Error uploading file: {e}")
		return {"status": "error"}
	finally:
		classification_file.file.close()
	if not allowed_company_bins:
		allowed_company_bins = DEFAULT_COMPANY_BINS_WITHOUT_COMPANIES
	df = prepare_data(classification_file.filename)
	conversation_data = create_hierarchical_data_prod(df)

	for conversation in tqdm(conversation_data):
		classification_result = classify_conversation(conversation, model=fine_tuned_model, company_bins=allowed_company_bins)
		conversation.update({"bin": classification_result})

	df_conversation = pd.DataFrame(conversation_data)
	df_conversation = df_conversation.explode(["conversations", "tweet_id"])
	df = df.merge(df_conversation[["tweet_id", "bin"]], on="tweet_id")
	df.to_csv("classification_results.csv", index=False)
	return FileResponse("classification_results.csv")

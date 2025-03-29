from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile
from quarter_lib.logging import setup_logging
from starlette.responses import FileResponse
from tqdm import tqdm

from src.config import DEFAULT_COMPANY_BINS_WITHOUT_COMPANIES
from src.services.data_service import create_hierarchical_data_prod, prepare_data, prepare_file, read_json, slugify
from src.services.openai_service import classify_conversation

logger = setup_logging(__name__)
router = APIRouter(tags=["classification"])


@router.post("/classify")
async def upload_file(
	classification_file: Annotated[UploadFile, File()],
	allowed_company_bins: Annotated[UploadFile, Form()] = None,
	fine_tuned_model: Annotated[str, Form()] = None,
):
	error_classification_file = prepare_file(classification_file)
	if error_classification_file:
		return {"error": "Invalid file format. Please upload a CSV file."}
	error_company_bins = prepare_file(allowed_company_bins)

	if error_company_bins:
		allowed_company_bins = DEFAULT_COMPANY_BINS_WITHOUT_COMPANIES
	else:
		allowed_company_bins = read_json(allowed_company_bins.filename)

	df = prepare_data(classification_file.filename)
	conversation_data = create_hierarchical_data_prod(df)

	for conversation in tqdm(conversation_data, desc="Classifying conversations"):
		classification_result = classify_conversation(conversation, model=fine_tuned_model, company_bins=allowed_company_bins)
		conversation.update({"bin": classification_result})

	df_conversation = pd.DataFrame(conversation_data)
	df_conversation = df_conversation.explode(["conversations", "tweet_id"])
	df = df.merge(df_conversation[["tweet_id", "bin"]], on="tweet_id")
	result_file_name = slugify(fine_tuned_model) + "_classification_results.csv"
	df.to_csv(result_file_name, index=False)
	return FileResponse(result_file_name)

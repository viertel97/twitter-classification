from typing import Annotated

import pandas as pd
from fastapi import APIRouter
from fastapi import File, UploadFile
from quarter_lib.logging import setup_logging
from starlette.responses import FileResponse
from tqdm import tqdm

from src.services.anonymization_service import update_text
from src.services.data_service import prepare_data

logger = setup_logging(__name__)
tqdm.pandas()

router = APIRouter()

@router.post("/anonymize")
async def anonymize(file_to_anonymize: Annotated[UploadFile, File]):
    try:
        contents = file_to_anonymize.file.read()
        with open(file_to_anonymize.filename, 'wb') as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {"status": "error"}
    finally:
        file_to_anonymize.file.close()
    df = prepare_data(file_to_anonymize.filename)
    # change dataframe to list of dictionaries
    df_list = df.to_dict(orient="records")
    for row in tqdm(df_list):
        row["text"] = update_text(row["text"])
    df = pd.DataFrame(df_list)
    df.to_csv("anonymized_data.csv")
    return FileResponse("anonymized_data.csv")
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI
from quarter_lib.logging import setup_logging

from config.api_documentation import description, tags_metadata, title
from src.controller import classification_controller, training_controller, anonymization_controller

controllers = [classification_controller, training_controller, anonymization_controller]

logger = setup_logging(__name__)


app = FastAPI(debug=False, openapi_tags=tags_metadata, title=title, description=description)
router = APIRouter()

[app.include_router(controller.router) for controller in controllers]


@app.get("/")
def health():
	return {"status": "ok"}


if __name__ == "__main__":
	uvicorn.run(f"{Path(__file__).stem}:app", host="0.0.0.0", reload=False, port=80)

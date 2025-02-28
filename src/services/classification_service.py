import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from quarter_lib.logging import setup_logging
from pydantic import BaseModel

from src.config import SEED

logger = setup_logging(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_CUSTOM_ENDPOINT = os.getenv("AZURE_CUSTOM_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")

client = AzureOpenAI(api_key=OPENAI_API_KEY, azure_endpoint=AZURE_CUSTOM_ENDPOINT, api_version="2024-08-01-preview")


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
        model=model,
        seed=SEED,
        messages=messages,
        functions=[function_schema],
        function_call={"name": "classify_company"}
    )

    message = response.choices[0].message

    if message.function_call:
        try:
            arguments = json.loads(message.function_call.arguments)
            classification_result = ClassificationResult(**arguments)
            return classification_result
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error parsing function call arguments: {e}")
            return None
    else:
        content = message.content
        try:
            arguments = json.loads(content)
            classification_result = ClassificationResult(**arguments)
            return classification_result
        except json.JSONDecodeError:
            logger.error("Unable to parse response content as JSON.")
            return None

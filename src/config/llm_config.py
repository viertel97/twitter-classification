import json


def get_function_schema(allowed_companies) -> dict:
	function_schema = {
		"name": "classify_company",
		"description": "Classify the company referred to in the tweet conversation.",
		"parameters": {
			"type": "object",
			"properties": {
				"company": {
					"type": "string",
					"enum": allowed_companies,
					"description": "The company name that best fits the conversation, or 'Unclear' if none applies.",
				}
			},
			"required": ["company"],
		},
	}
	return function_schema


def get_messages(conversation_data) -> list:
	return [
		{
			"role": "user",
			"content": f"Please classify the following tweet conversation to one of the allowed companies: {json.dumps(conversation_data)}",
		}
	]

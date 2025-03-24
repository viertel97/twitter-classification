import json


def get_classification_function_schema(allowed_companies) -> dict:
	return {
		"name": "classify_company",
		"description": "Assign the most appropriate industry bin based on the context of the tweet conversation.",
		"parameters": {
			"type": "object",
			"properties": {
				"bin": {
					"type": "string",
					"enum": allowed_companies,
					"description": "The industry bin that best fits the conversation context.",
				}
			},
			"required": ["bin"],
		},
	}

def get_classification_prompt(allowed_bins: dict, conversation_data: list) -> list:
	return [
        {
            "role": "user",
            "content":
                f"""You are provided with two pieces of information:
                1. A dictionary mapping industry bins to allowed companies. Each key is an industry bin and its value is a list of companies belonging to that bin:
                {json.dumps(allowed_bins, indent=2)}
 
                2. A list of tweets from customers of a company - the company responses are not included:
                {json.dumps(conversation_data, indent=2)}
                Analyze the customer tweets and determine which industry bin from the dictionary best fits the context of these tweets. Return only the name of the selected bin (i.e., the key).
                If there is any indication of an industry, choose the corresponding bin. Only if the tweets have no relation to any of the provided industries should you return 'None'.
				"""
    	}
	]


def get_binning_prompt(companies: set[str]) -> list:
	return [{"role": "user", "content": f"Classify the following companies into industry-based groups. Use your knowledge to determine appropriate industry categories, considering common business sectors rather than just company names. The level of granularity is flexible but should maintain meaningful distinctions. Return the result as a dictionary / json, where the keys are industry names and the values are lists of companies belonging to that industry. Companies: {json.dumps(companies)}"}]
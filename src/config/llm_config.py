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
				},
			},
			"required": ["bin"]
		},
	}


def get_classification_prompt(allowed_bins: list[dict], conversation_data: list) -> list:
	return [
		{
			"role": "user",
			"content": f"""
You are provided with two sets of data:

1. A list of dictionaries representing industry bins. Each dictionary contains:
   - The name of the industry bin.
   - Relevant keywords for context.

Here is the data for industry bins:
{json.dumps(allowed_bins, indent=2)}

2. A list of customer tweets (company responses are not included), arranged in chronological order.

Here is the conversation data:
{json.dumps(conversation_data, indent=2)}

Your task is to:
- Analyze the entire tweet conversation.
- Identify the industry bin that best fits the overall context of the tweets.
- Consider the keywords associated with each bin. A keyword match in the tweets should strongly indicate the appropriate bin.
- If any tweet explicitly indicates an industry, select that bin.
- If none of the tweets clearly relate to any industry in the list, return "None".

Your output should include:
- bin: The name of the chosen industry bin (do not return a company name).
""",
		}
	]


def get_description_prompt(bin_name: str) -> str:
	return [
		{
			"role": "user",
			"content": f"""
You are provided with the name of an industry bin. Your task is:
- Creating a list of 5-10 keywords that are relevant to the industry bin.

Industry bin: {bin_name}

Return your answer in a json format with the key "keywords".
""",
		}
	]


def get_binning_prompt(company: str) -> str:
	return [
		{
			"role": "user",
			"content": f"""
Using your external knowledge about each company, classify the following company into one of the following industry-based groups based on the primary industry of the company they represent. For example, assign askPayPal to Financial Services (since PayPal is a financial services company) and askEbay to Retail (since eBay is a retail company).
The possible industry bins are:
- Entertainment & Media
- Finance & Banking
- Telecommunications
- Travel & Hospitality
- Technology & Software
- Retail & Food Services
- Healthcare & Pharmaceuticals
- Transportation & Logistics

Return the name of the bin from the list above in a json format with the key "bin".

Company to classify: {company}
""",
		}
	]

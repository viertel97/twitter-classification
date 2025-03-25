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
				"reasoning": {
					"type": "string",
					"description": "Explain why the selected bin is the most appropriate choice.",
				},
			},
			"required": ["bin", "reasoning"],  # This must be at the same level as "properties"
		},
	}


def get_classification_prompt(allowed_bins: list[dict], conversation_data: list) -> list:
	return [
		{
			"role": "user",
			"content": f"""
You are given two sets of information:

1. A list of dictionaries containing the name of the industry bin, their allowed companies and some keywords related to that industry for context:
{json.dumps(allowed_bins, indent=2)}

2. A list of customer tweets (company responses are not included):
{json.dumps(conversation_data, indent=2)}

Your task is to:
- Read all the tweets as one complete conversation - the tweets are in chronological order and the tweets from the customer agents are not included.
- Identify which industry bin from the dictionary best fits the overall context of the tweets - also consider the keywords associated with each bin. If you find a keyword that in the conversation, it should be a strong indicator of the correct bin.
- Use the companies associated with each bin as guidance.
- If any tweet clearly indicates an industry, choose that bin.
- If none of the tweets relate to any industry in the dictionary, return 'None'.

Your output should include:
- bin: The name of the chosen industry bin. NOT the name of the company.
- explanation: A brief explanation of why you selected this bin.

Return your answer as: [bin, explanation]
""",
		}
	]


def get_binning_prompt(companies: set[str]) -> list[dict]:
	return [
		{
			"role": "user",
			"content": f"""
"Using your external knowledge about each company, classify the following companies into industry-based groups based on the primary industry of the company they represent. For example, place askPayPal under Financial Services (since PayPal is a financial services company) and askEbay under Retail (since eBay is a retail company). Do not group any accounts which contain 'ask', 'help', 'care' or 'support' under a generic customer support or care bin—instead, assign them to their proper company's industry.

For each industry bin, generate a distinct list of 15 indicative keywords based on the characteristics of that industry. Ensure the keywords are specific and distinct enough to differentiate between industries. Especially focus on the unique aspects of each industry and the words that are most likely to appear in tweets from customers of companies in that industry and make them as unique as possible.

Also, ensure that no resulting industry bin contains less then 3 companies. If a bin would contain just one company, merge it with a similar or more general category.

Return the result as a list of dictionaries / nested json, where each dictionary contains:
- 'bin': the name of the industry bin
- 'keywords': a list of indicative keywords for that bin
- 'companies': a list of companies belonging to that industry

Companies: {json.dumps(companies)}"
""",
		}
	]

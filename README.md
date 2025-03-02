# twitter-classification

## Description
This project is a simple script that uses the OpenAI API to classify tweets.
It has the following functionality:
- Anonymize the tweet-data
- Fine-tune a model with the anonymized data
- Classify the tweets using the trained model
## Installation

## Usage

### Configuration
- Create a `.env` file in the root directory of the project - it needs to contain the following variable:
```
OPENAI_API_KEY=XYZ
```

1. Clone the repository
2. Install the requirements
3. Run the script

## Further improvements
- Adaption the anonymization method - the current one (Presidio Analyzer) creates reuses the same anonymization for the same name. Therefore, "John" will always be <PERSON_0> and "Jane" will always be <PERSON_1>. Maybe replacing the original names with fake names would be a better approach.
- Automatically detect language of the tweet and switch model accordingly
- Hyperparameter tuning
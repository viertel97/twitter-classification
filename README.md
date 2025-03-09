# twitter-classification

## Description
This project is a simple script that uses the OpenAI API to classify tweets.
It has the following functionality:
- Anonymize the tweet-data
- Fine-tune a model with the anonymized data
- Gather job data from the OpenAI API
- Classify the tweets using the trained model
## Installation

## Usage

### Configuration
- Create a `.env` file in the root directory of the project - it needs to contain the following variable:
```
OPENAI_API_KEY=XYZ
```
or

```
AZURE_OPENAI_API_KEY=XYZ
AZURE_ENDPOINT=XYZ

```

1. Clone the repository
2. Populate the `.env` file with your OpenAI API key
3. Run `docker build -t foo . && docker run --env-file .env -it foo 
4. Check 

## Further improvements
- Adaption the anonymization method - the current one (Presidio Analyzer) creates reuses the same anonymization for the same name. Therefore, "John" will always be <PERSON_0> and "Jane" will always be <PERSON_1>. Maybe replacing the original names with fake names would be a better approach.
- Automatically detect language of the tweet and switch NER model accordingly.
- Hyperparameter tuning of LLM fine-tuning.
- Using LLM evaluation framework to evaluate and test the model (eg. DeepEval).
- Tracking token usage via code (saving the token usage per request / per training).
- Using different fine-tuning methods instead of supervised eg. [DPO](https://arxiv.org/abs/2305.18290).
- Using the same batch_size and learning rate for both fine-tuning models to make them more comparable.

## Known issues
- Due to problems with [cython-blis](https://github.com/explosion/cython-blis/issues/117) (a spacy dependency) I added the following line to the Dockerfile:
```
BLIS_ARCH=generic
```
This is a workaround and should be removed as soon as the issue is fixed.
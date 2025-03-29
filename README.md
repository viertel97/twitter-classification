# twitter-classification

## Description
This project is a simple script that uses the (Azure) OpenAI API to classify tweets.
It has the following functionality:
- Anonymize the tweet-data
- Fine-tune a model with the anonymized data
- Generate bins based on the industry of the company
- Gather job data from the OpenAI API
- Classify the tweets using the trained model

## Configuration
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
3. Run `docker build -t foo . && docker run --env-file .env -it foo`

## Usage
1. Anonymize - use the `/anonymize` endpoint and save the file-result
2. Optional: Group companies - use the `/group_companies` endpoint and save the file-result
3. Fine-tune the model - use the `/train` endpoint and save the job response
4. Check if the model is ready - use the `/get_status` endpoint with the job_id from the previous step
5. Classify the tweets - use the `/classify` endpoint and use the model_name and file tweets to classify. Optionally use the file-result from step 2, otherwise use the default will be used.


## Further improvements
- Adaption the anonymization method - the current one (Presidio Analyzer) creates reuses the same anonymization for the same name. Therefore, "John" will always be <PERSON_0> and "Jane" will always be <PERSON_1>. Maybe replacing the original names with fake names would be a better approach.
- Automatically detect language of the tweet and switch NER model accordingly.
- Hyperparameter tuning of LLM fine-tuning.
- Using LLM evaluation framework to evaluate and test the model (eg. DeepEval).
- Tracking token usage via code (saving the token usage per request / per training).
- Using different fine-tuning methods instead of supervised eg. [DPO](https://arxiv.org/abs/2305.18290).
- Using the same batch_size and learning rate for fine-tuning to make the tunings them more comparable.
- Add more / less possible industries for the binning-prompt.
- Create agent system which validates the output and checks if the output is correct.

## Known issues
- Due to problems with [cython-blis](https://github.com/explosion/cython-blis/issues/117) (a spacy dependency) I added the following line to the Dockerfile:
```
BLIS_ARCH=generic
```
This is a workaround and should be removed as soon as the issue is fixed.
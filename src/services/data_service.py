import json

import numpy as np
import pandas as pd
from quarter_lib.logging import setup_logging

logger = setup_logging(__name__)

def find_root_tweet_id(tweet_id, parent_map):
    current = tweet_id
    visited = set()

    while True:
        parent = parent_map.get(current, np.nan)
        if pd.isna(parent) or parent in visited:
            return current
        visited.add(current)
        current = parent

def generate_hierarchy(df):
    parent_map = dict(zip(df['tweet_id'], df['in_response_to_tweet_id']))
    df['conversation_id'] = df['tweet_id'].apply(lambda t: find_root_tweet_id(t, parent_map))

    return df

def create_hierarchical_data(df):
    df = generate_hierarchy(df)
    grouped = df.groupby('conversation_id')

    conversation_data, companies = [], []
    for conv_id, group in grouped:
        authors = group[group["inbound"] == False]['author_id'].unique()
        if not len(
                authors) == 1:  # in some conversations are more than just one named author, which will be removed here
            continue
        else:
            company = authors[0]
        group = group[group["inbound"] == True]
        group = group.sort_values(by='created_at')

        group['text'] = group['text'].str.replace(company, '', case=False)

        conversation_data.append({
            'conversations': group["text"].to_list(),
            'company': company,
            'tweet_id': group['tweet_id'].to_list()
        })
        companies.append(company)
    logger.info(f"Number of conversations: {len(conversation_data)}")
    return conversation_data, set(companies)


def create_hierarchical_data_prod(df):
    df = generate_hierarchy(df)
    grouped = df.groupby('conversation_id')

    conversation_data = []
    for conv_id, group in grouped:
        conversation_data.append({
            'conversations': group["text"].to_list(),
            'tweet_id': group['tweet_id'].to_list()
        })
    return conversation_data


def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df['tweet_id'].apply(lambda x: str(x).isdigit())]
    df = df[df['text'].apply(lambda x: isinstance(x, str))]
    df['tweet_id'] = df['tweet_id'].astype(int)
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], format="%a %b %d %H:%M:%S %z %Y")
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].astype(float)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'response_tweet_id' in df.columns:
        df.drop(columns=['response_tweet_id'], inplace=True)
    return df

def save_to_jsonl(data, output_file_path):
    jsonl_data = []
    for row in data:
        jsonl_data.append({
            "messages": [
                {"role": "system",
                 "content": "Your purpose is to classify Twitter posts and decide which company this support request belongs to. Different incoming user messages from one conversation are split by a '~'"},
                {"role": "user", "content": " ~ ".join(row['conversations'])},
                {"role": "assistant", "content": f"\"{row['company']}\""}
            ]
        })

    with open(output_file_path, 'w+') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')

    logger.info(f"Saved data to {output_file_path}")
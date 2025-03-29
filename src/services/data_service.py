import json

import numpy as np
import pandas as pd
from fastapi import UploadFile
from quarter_lib.logging import setup_logging
from tqdm import tqdm

from src.services.openai_service import describe_bin, group_company

logger = setup_logging(__name__)


def find_root_tweet_id(tweet_id: int, parent_map: dict[int, int]) -> int:
	current = tweet_id
	visited = set()

	while True:
		parent = parent_map.get(current, np.nan)
		if pd.isna(parent) or parent in visited:
			return current
		visited.add(current)
		current = parent


def generate_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
	parent_map = dict(zip(df["tweet_id"], df["in_response_to_tweet_id"], strict=False))
	df["conversation_id"] = df["tweet_id"].apply(lambda t: find_root_tweet_id(t, parent_map))

	return df


def create_hierarchical_data(df) -> tuple[list[dict[str, list[str]]], set[str]]:
	df = generate_hierarchy(df)
	grouped = df.groupby("conversation_id")

	conversation_data, companies = [], []
	for conv_id, group in tqdm(grouped, desc="Creating hierarchical data from conversations"):
		authors = group[group["inbound"] == False]["author_id"].unique()
		if not len(authors) == 1:  # in some conversations are more than just one named author, which will be removed here
			continue
		company = authors[0]
		group = group[group["inbound"] == True]
		group = group.sort_values(by="created_at")

		group["text"] = group["text"].str.replace(company, "", case=False)

		conversation_data.append({"conversations": group["text"].to_list(), "company": company, "tweet_id": group["tweet_id"].to_list()})
		companies.append(company)
	logger.info(f"Number of conversations: {len(conversation_data)}")
	return conversation_data, set(companies)


def create_hierarchical_data_prod(df: pd.DataFrame) -> list[dict]:
	df = generate_hierarchy(df)
	grouped = df.groupby("conversation_id")

	conversation_data = []
	for conv_id, group in tqdm(grouped, desc="Creating hierarchical data from conversations (production)"):
		conversation_data.append({"conversations": group["text"].to_list(), "tweet_id": group["tweet_id"].to_list()})
	return conversation_data


def prepare_data(file_path: str) -> pd.DataFrame:
	df = pd.read_csv(file_path)
	df = df[df["tweet_id"].apply(lambda x: str(x).isdigit())]
	df = df[df["text"].apply(lambda x: isinstance(x, str))]
	df["tweet_id"] = df["tweet_id"].astype(int)
	if "created_at" in df.columns and df["created_at"].dtype == "datetime64[ns]":
		df["created_at"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y")
	df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype(float)
	if "Unnamed: 0" in df.columns:
		df.drop(columns=["Unnamed: 0"], inplace=True)
	if "response_tweet_id" in df.columns:
		df.drop(columns=["response_tweet_id"], inplace=True)
	return df


def save_to_jsonl(data: list[dict[str, list[str]]], output_file_path: str) -> None:
	jsonl_data = []
	for row in data:
		jsonl_data.append(
			{
				"messages": [
					{
						"role": "system",
						"content": "Your purpose is to classify Twitter posts and decide which company this support request belongs to. Different incoming user messages from one conversation are split by a '~'",
					},
					{"role": "user", "content": " ~ ".join(row["conversations"])},
					{"role": "assistant", "content": f'"{row["company"]}"'},
				]
			}
		)

	with open(output_file_path, "w+") as f:
		for item in jsonl_data:
			f.write(json.dumps(item) + "\n")

	logger.info(f"Saved data to {output_file_path}")


def prepare_file(upload_file: UploadFile) -> dict | None:
	try:
		contents = upload_file.file.read()
		with open(upload_file.filename, "wb") as f:
			f.write(contents)
	except Exception as e:
		logger.error(f"Error uploading file: {e}")
		return {"status": "error"}
	finally:
		upload_file.file.close()
	return None


def read_json(file_path: str) -> list[dict]:
	with open(file_path, "r") as f:
		data = json.load(f)
	return data


def create_company_bins(companies: list[str]) -> list[dict]:
	bin_dict = {
		"Entertainment & Media": [],
		"Finance & Banking": [],
		"Telecommunications": [],
		"Travel & Hospitality": [],
		"Technology & Software": [],
		"Retail & Food Services": [],
		"Healthcare & Pharmaceuticals": [],
		"Transportation & Logistics": [],
	}
	for company in companies:
		res = group_company(company)
		logger.info(f"{company}: {res}")
		bin_dict[res].append(company)

	bin_list = []
	for company_bin in bin_dict:
		bin_list.append({"bin": company_bin, "companies": bin_dict[company_bin]})

	for company_bin in bin_list:
		keywords = describe_bin(company_bin["bin"])
		logger.info(f"{company_bin['bin']}: {keywords}")
		if keywords is not None:
			company_bin["keywords"] = keywords
	bin_list.append({"bin": "None", "companies": [], "keywords": []})
	return bin_list


import unicodedata
import re


def slugify(value, allow_unicode=False):
	value = str(value)
	if allow_unicode:
		value = unicodedata.normalize("NFKC", value)
	else:
		value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
	value = re.sub(r"[^\w\s-]", "", value.lower())
	return re.sub(r"[-\s]+", "-", value).strip("-_")

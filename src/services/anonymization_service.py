import re
from typing import Dict

import spacy
from dotenv import load_dotenv
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer import OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType
from quarter_lib.logging import setup_logging

logger = setup_logging(__name__)

load_dotenv()

MODEL_CONFIG = [{"lang_code": "en", "model_name": {"spacy": "en_core_web_trf", "transformers": "dslim/bert-base-NER"}}]


try:
	NLP_ENGINE = TransformersNlpEngine(models=MODEL_CONFIG)
except Exception as e:
	logger.error(f"Error loading NLP engine: {e}")
	logger.info("Downloading spacy model")
	spacy.cli.download("en_core_web_trf")
	NLP_ENGINE = TransformersNlpEngine(models=MODEL_CONFIG)

ANALYZER = AnalyzerEngine(nlp_engine=NLP_ENGINE)

ANONYMIZER_ENGINE = AnonymizerEngine()


class InstanceCounterAnonymizer(Operator):
	REPLACING_FORMAT = "<{entity_type}_{index}>"

	def operate(self, text: str, params: Dict = None) -> str:
		"""Anonymize the input text."""

		entity_type: str = params["entity_type"]

		# entity_mapping is a dict of dicts containing mappings per entity type
		entity_mapping: Dict[Dict:str] = params["entity_mapping"]

		entity_mapping_for_type = entity_mapping.get(entity_type)
		if not entity_mapping_for_type:
			new_text = self.REPLACING_FORMAT.format(entity_type=entity_type, index=0)
			entity_mapping[entity_type] = {}

		else:
			if text in entity_mapping_for_type:
				return entity_mapping_for_type[text]

			previous_index = self._get_last_index(entity_mapping_for_type)
			new_text = self.REPLACING_FORMAT.format(entity_type=entity_type, index=previous_index + 1)

		entity_mapping[entity_type][text] = new_text
		return new_text

	@staticmethod
	def _get_last_index(entity_mapping_for_type: Dict) -> int:
		"""Get the last index for a given entity type."""

		def get_index(value: str) -> int:
			return int(value.split("_")[-1][:-1])

		indices = [get_index(v) for v in entity_mapping_for_type.values()]
		return max(indices)

	def validate(self, params: Dict = None) -> None:
		"""Validate operator parameters."""

		if "entity_mapping" not in params:
			raise ValueError("An input Dict called `entity_mapping` is required.")
		if "entity_type" not in params:
			raise ValueError("An entity_type param is required.")

	def operator_name(self) -> str:
		return "entity_counter"

	def operator_type(self) -> OperatorType:
		return OperatorType.Anonymize


ANONYMIZER_ENGINE.add_anonymizer(InstanceCounterAnonymizer)
ENTITY_MAPPING = dict()


def clean_text(text: str) -> str:
	cleaned_text = remove_mentions(text).strip()
	return anonymize_text(cleaned_text)


def remove_mentions(text: str) -> str:
	return re.sub(r"@[\w]{3,16}", "", text)


def anonymize_text(text: str) -> str:
	analyzer_results = ANALYZER.analyze(text=text, language="en")

	anonymized_result = ANONYMIZER_ENGINE.anonymize(
		text,
		analyzer_results,
		{"DEFAULT": OperatorConfig("entity_counter", {"entity_mapping": ENTITY_MAPPING})},
	)

	return anonymized_result.text

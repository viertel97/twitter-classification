import os
from pathlib import Path

SEED = 1337

DATA_PATH = os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, 'data')

training_file_name = 'training.jsonl'
validation_file_name = 'validation.jsonl'

training_file_path = os.path.join(DATA_PATH, training_file_name)
validation_file_path = os.path.join(DATA_PATH, validation_file_name)

print(DATA_PATH)
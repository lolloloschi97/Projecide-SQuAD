import numpy as np
import pandas as pd
import pickle

EMBEDDING_DIM = 200
UTILS_ROOT = "../utils/"
DATASET_ROOT = "../datasets/"
INPUT_FILE_NAME = "raw_dataset"
TRAIN_SIZE = 0.8

RAW_POS_LIST = {'ADJ': 4, 'ADP': 0, 'ADV': 4, 'AUX': 0, 'CONJ': 0, 'CCONJ': 0, 'DET': 0, 'INTJ': 0, 'NOUN': 2, 'NUM': 1, 'PART': 0, 'PRON': 0, 'PROPN': 1, 'PUNCT': 0, 'SCONJ': 0, 'SYM': 0, 'VERB': 3, 'X': 0}

POS_LIST = {'ADJ': 4, 'ADV': 4, 'NOUN': 2, 'NUM': 1, 'PROPN': 1, 'VERB': 3}





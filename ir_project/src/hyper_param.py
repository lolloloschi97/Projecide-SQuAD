import numpy as np
import pandas as pd
import pickle

EMBEDDING_DIM = 200
UTILS_ROOT = "../utils/"
DATASET_ROOT = "../datasets/"
INPUT_FILE_NAME = "raw_dataset"
TRAIN_SIZE = 0.8

RAW_POS_LIST = {'CC': 0, 'CD': 5, 'DT': 0, 'EX': 0, 'FW': 6, 'IN': 0, 'JJ': 4, 'JJR': 4, 'JJS': 4, 'LS': 0, 'MD': 0, 'NN': 2, 'NNS': 2, 'NNP': 1, 'NNPS': 1, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$':0, 'RB': 4, 'RBR': 4, 'RBS': 4, 'RP': 0, 'TO': 0, 'UH': 0, 'VB': 3, 'VBD': 3, 'VBG': 3, 'VBN': 3, 'VBP': 3, 'VBZ': 3, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0}

POS_LIST = {'CD': 5, 'FW': 6, 'JJ': 4, 'JJR': 4, 'JJS': 4, 'NN': 2, 'NNS': 2, 'NNP': 1, 'NNPS': 1, 'RB': 4, 'RBR': 4, 'RBS': 4, 'VB': 3, 'VBD': 3, 'VBG': 3, 'VBN': 3, 'VBP': 3, 'VBZ': 3}





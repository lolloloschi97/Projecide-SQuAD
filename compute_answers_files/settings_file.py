import pandas as pd
import numpy as np
import json
import os
import pickle
import tensorflow as tf
import tqdm
import copy


MAX_LENGTH_CONTEXT = 501  # from pre-trained model
MAX_LENGTH_QUESTION = 40  # from pre-trained model
NUM_POS_TOKENS = 38  # from pre-trained model
INPUT_FILE_FOLDER = "\\predictions\\in\\"
OUTPUT_FILE_FOLDER = "\\predictions\\out\\"
UTILS_ROOT = "\\utils\\"

import pandas as pd
import numpy as np
import json
import os
import pickle
import tensorflow as tf

MAX_SEQ_LENGTH_X_CONTEXT = 501  # from pre-trained model
NUM_POS_TOKENS = 38  # from pre-trained model
INPUT_FILE_FOLDER = "\\compute_answers_files\\predictions\\in\\"
OUTPUT_FILE_FOLDER = "\\compute_answers_files\\predictions\\out\\"
UTILS_ROOT = "\\compute_answers_files\\utils\\"

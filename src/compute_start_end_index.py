import pandas as pd
import numpy as np
from hyper_param import *


def compute_start_end_index(training_df, validation_df):
    print("Training set")
    context_answer_df = training_df.iloc[:][['context', 'text']]
    start_indexes = []
    end_indexes = []
    print("Start computing indexes...")
    for i, row in context_answer_df.iterrows():
        answer = row.text
        context = row.context
        start = len(context[:context.find(answer)].split(' '))  # count the words before the match
        end = start - 1 + len(answer.split(' '))
        start_indexes.append(start)
        end_indexes.append(end)
        if start == -1:
            print(i)    # check if, for any reason, context does not contain the answer
    print("Indexes computed")
    new_training_df = training_df[['id', 'question', 'context', 'text', 'context_id']]
    new_training_df['start_index'] = start_indexes
    new_training_df['end_index'] = end_indexes
    new_training_df.to_csv(DATASET_ROOT + 'training.csv')

    print("")
    print("Validation set")
    context_answer_df = validation_df.iloc[:][['context', 'text']]
    start_indexes = []
    end_indexes = []
    print("Start computing indexes...")
    for i, row in context_answer_df.iterrows():
        answer = row.text
        context = row.context
        start = len(context[:context.find(answer)].split(' '))
        end = start - 1 + len(answer.split(' '))
        start_indexes.append(start)
        end_indexes.append(end)
        if start == -1:
            print(i)    # check if, for any reason, context does not contain the answer
    print("Indexes computed")
    new_validation_df = validation_df[['id', 'question', 'context', 'text', 'context_id']]
    new_validation_df['start_index'] = start_indexes
    new_validation_df['end_index'] = end_indexes
    new_training_df.to_csv(DATASET_ROOT + 'validation.csv')

    return new_training_df, new_validation_df



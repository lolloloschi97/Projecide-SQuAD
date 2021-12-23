import json
import numpy as np
import pickle
import pandas as pd


def squad_json_to_dataframe_train(input_file):
    """
    :param input_file: dataset name
    :return: Pandas dataframe of dataset
    """

    print("Reading the json file")
    file = json.loads(open("../datasets/%s.json" % input_file).read())
    print("processing...")

    # parsing different level's in the json file
    nested_tags = ['data', 'paragraphs', 'qas', 'answers']
    answers_df = pd.json_normalize(file, nested_tags)  # each 'answers' consists of 'answer_start', 'text'
    qas_df = pd.json_normalize(file, nested_tags[:-1])  # each 'qas' consists of 'answers', 'question', 'id'
    paragraph_df = pd.json_normalize(file, nested_tags[:-2])  # each 'paragraphs' consists of 'context' and 'qas'

    #  combining it into single dataframe
    idx = np.repeat(paragraph_df['context'].values, paragraph_df['qas'].str.len())
    ndx = np.repeat(qas_df['id'].values, qas_df['answers'].str.len())
    qas_df['context'] = idx
    answers_df['q_idx'] = ndx
    dataframe = pd.concat([qas_df[['id', 'question', 'context']].set_index('id'), answers_df.set_index('q_idx')], 1, sort=False).reset_index()
    dataframe['c_id'] = dataframe['context'].factorize()[0]
    print("shape of the dataframe is {}".format(dataframe.shape))
    return dataframe


input_file = "raw_dataset"
pd_dataframe = squad_json_to_dataframe_train(input_file=input_file)

pd_dataframe.to_csv("out.csv")

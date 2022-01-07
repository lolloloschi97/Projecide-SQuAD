import json
from hyper_param import *


def json_to_dataframe(input_file):
    """
    :param input_file: dataset name
    :return: Pandas dataframe of dataset with columns: 'index', 'question', 'context', 'answer_start, 'text', 'context_id'
    """

    print("Reading the json file")
    file = json.loads(open(DATASET_ROOT + "%s.json" % input_file).read())
    print("processing...")

    # parsing different level's in the json file
    nested_tags = ['data', 'paragraphs', 'qas', 'answers']
    answers_df = pd.json_normalize(file, nested_tags)  # each 'answers' consists of 'answer_start', 'text'
    qas_df = pd.json_normalize(file, nested_tags[:-1])  # each 'qas' consists of 'answers', 'question', 'index'
    paragraph_df = pd.json_normalize(file, nested_tags[:-2])  # each 'paragraphs' consists of 'context' and 'qas'

    #  combining it into single dataframe
    context_values_column = np.repeat(paragraph_df['context'].values, paragraph_df['qas'].str.len())  # duplicate a context many times as the number of questions
    qas_df['context'] = context_values_column  # add the 'context' column to each Q&A pair-> 'answers', 'question', 'index', 'context'
    answers_df['id'] = qas_df['id'].values  # add the same index of the qas_df to the Q&A pairs
    # indexes must be aligned before merging the columns
    dataframe = pd.concat([qas_df[['id', 'question', 'context']].set_index('id'), answers_df.set_index('id')], axis=1, sort=False).reset_index()  # merge the columns 'question', 'index', 'context' to 'answer_start', 'text'
    dataframe['context_id'] = dataframe['context'].factorize()[0]  # add a column to associate an integer to the same context
    print("shape of the dataframe is {}".format(dataframe.shape))
    return dataframe


def split_dataset(dataframe):
    training_max_context_id = int(TRAIN_SIZE * max(dataframe["context_id"]))
    val_max_context_id = int((1-TEST_SIZE) * max(dataframe["context_id"]))

    train_set = dataframe.loc[dataframe["context_id"] < training_max_context_id]
    val_set = dataframe.loc[dataframe["context_id"] >= training_max_context_id]
    val_set = val_set.loc[val_max_context_id >= val_set["context_id"]]
    test_set = dataframe.loc[dataframe["context_id"] > val_max_context_id]
    print("Datasets split")
    return train_set, val_set.reset_index(), test_set.reset_index()


# MAIN FUNCTION
def data_loader():
    """
    :return: Pandas dataframes of training set, validation set and test set
    """

    pd_dataframe = json_to_dataframe(INPUT_FILE_NAME)
    train_set, val_set, test_set = split_dataset(pd_dataframe)

    return train_set, val_set, test_set


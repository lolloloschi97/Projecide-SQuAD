import os

from settings_file import *


def json_to_dataframe(file_path):
    """
    :param input_file: dataset name
    :return: Pandas dataframe of dataset with columns: 'index', 'question', 'context', 'context_id'
    """
    print("Reading the json file")
    file = None
    opened = False
    try:
        print("Try to open as cp1252")
        file = json.loads(open(file_path, encoding="Windows-1252").read())
        opened = True
    except:
        print("File encoding is not CP1252")

    if not opened:
        try:
            print("Try to open as utf-8")
            file = json.loads(open(file_path, encoding="utf-8").read())
        except:
            print("File encoding is not UTF-8")
            raise

    print("SUCCESS! Processing...")

    # parsing different level's in the json file
    nested_tags = ['data', 'paragraphs', 'qas']
    qas_df = pd.json_normalize(file, nested_tags)  # each 'qas' consists of 'question', 'index'
    paragraph_df = pd.json_normalize(file, nested_tags[:-1])  # each 'paragraphs' consists of 'context' and 'qas'

    #  combining it into single dataframe
    context_values_column = np.repeat(paragraph_df['context'].values, paragraph_df[
        'qas'].str.len())  # duplicate a context many times as the number of questions
    qas_df[
        'context'] = context_values_column  # add the 'context' column to each question-> 'question', 'index', 'context'
    # indexes must be aligned before merging the columns
    dataframe = qas_df[['id', 'question', 'context']].set_index(
        'id').reset_index()  # merge the columns 'question', 'index', 'context'
    dataframe['context_id'] = dataframe['context'].factorize()[
        0]  # add a column to associate an integer to the same context
    print("shape of the dataframe is {}".format(dataframe.shape))
    return dataframe


def load_dataframe(file_path):
    cwd = os.getcwd()
    return json_to_dataframe(cwd + file_path)

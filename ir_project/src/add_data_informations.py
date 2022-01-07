import copy

from hyper_param import *
import nltk
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')


def add_exact_match(dataframe):
    print()
    print("Add exact match. It takes a while...")
    match_column = []
    for i, row in dataframe.iterrows():
        match_list = np.zeros(len(row.context.split(' ')))
        for k, c_word in enumerate(row.context.split(' ')):
            if c_word in row.question.split(' '):
                match_list[k] = 1
        match_column.append(list(match_list))
    dataframe['exact_match'] = pd.DataFrame({'exact_match': match_column})
    print("Exact match computed.")
    return dataframe

def add_data_informations(training_df, validation_df):
    training_df = add_exact_match(training_df)
    print()
    print("Repeat tagging for validation")
    validation_df = add_exact_match(validation_df)
    return training_df, validation_df

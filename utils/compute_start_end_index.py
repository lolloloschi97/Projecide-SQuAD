import pandas as pd


def compute_start_end_index(training_df, validation_df):
    context_answer_df = training_df.iloc[:][['context', 'text']]
    # for index, row in context_answer_df.iterrows():
    print((context_answer_df.iloc[0]['context']).find(context_answer_df.iloc[0]['text']))

from hyper_param import *

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def func(x):
    doc = nltk.pos_tag(nltk.word_tokenize(x))
    tag_list = dict((k, []) for k in POS_LIST.keys())
    for token in doc:
        tag = token[1]
        if tag in POS_LIST.keys():
            tag_list[tag].append(token[0])
    return tag_list


def document_tagging(training_df, validation_df):
    print("Document tagging...")
    training_df['document_tag'] = training_df['context']
    training_df.document_tag = training_df.document_tag.apply(lambda x: func(x))
    training_df.to_csv("out.csv")
    return training_df, validation_df

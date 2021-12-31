import pip

from hyper_param import *

#import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#from nltk import pos_tag
#from nltk import word_tokenize

import spacy
# spacy.cli.download("en_core_web_lg")
nlp_spacy = spacy.load('en_core_web_lg', exclude=["ner", "parser", "entity_linker", "entity_ruler", "textcat", "textcat_multilabel", "morphologizer", "senter", "sentencizer", "tok2vec", "transformer"])


def func(x):
    doc = nlp_spacy(x)
    tag_list = dict((k, []) for k in POS_LIST.keys())
    for token in doc:
        tag = token.pos_
        if tag in POS_LIST.keys():
            tag_list[tag].append(token.text)
    return tag_list


def document_tagging(training_df, validation_df):
    print("Document tagging...")
    training_df['document_tag'] = training_df['context']
    training_df.document_tag = training_df.document_tag.apply(lambda x: func(x))
    training_df.to_csv("out.csv")
    return training_df, validation_df

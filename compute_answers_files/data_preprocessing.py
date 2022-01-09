import re
from functools import reduce
import nltk
from nltk.corpus import stopwords

# Config

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@;\']')
GOOD_SYMBOLS_RE = re.compile('[^0-9a-zA-Zèé #+_]')
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))


def replace_special_characters(text: str) -> str:
    """
    Replaces special characters, such as paranthesis,
    with spacing character
    """
    text = re.sub(r"([0-9]+),([0-9]+)", r"\1\2", text)
    text = re.sub(r"([0-9]+(\.[0-9]+)?)", r" \1 ", text).strip()
    return REPLACE_BY_SPACE_RE.sub(' ', text)


def filter_out_uncommon_symbols(text: str) -> str:
    """
    Removes any special character that is not in the
    good symbols list (check regular expression)
    """
    return GOOD_SYMBOLS_RE.sub('', text)


def strip_text(text: str) -> str:
    """
    Removes any left or right spacing (including carriage return) from text.
    Example:
    Input: '  This assignment is cool\n'
    Output: 'This assignment is cool'
    """
    return text.strip()


def remove_redundant_spaces(text: str) -> str:
    """
    Removes redundant spaces
    """
    return re.sub(r" +", ' ', text)


PREPROCESSING_PIPELINE = [
                          replace_special_characters,
                          filter_out_uncommon_symbols,
                          strip_text,
                          remove_redundant_spaces
                          ]


# Anchor method
def text_prepare(text: str, filter_methods=None):
    """
    Applies a list of pre-processing functions in sequence (reduce).
    Note that the order is important here!
    """
    filter_methods = filter_methods if filter_methods is not None else PREPROCESSING_PIPELINE
    return reduce(lambda txt, f: f(txt), filter_methods, text)


# MAIN FUNCTION
def data_preprocessing(input_df):
    print('Pre-processing text...')
    print()
    print('[Debug] Before:\n{}'.format(input_df.context[:3]))
    print()

    for label in ['question', 'context']:
        input_df[label] = input_df[label].apply(lambda txt: text_prepare(txt))

    print('[Debug] After:\n{}'.format(input_df.context[:3]))
    print()
    print("Pre-processing completed!")

    return input_df

import re
from functools import reduce
import nltk
import copy
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


def lower(text: str) -> str:
    """
    Transforms given text to lower case.
    Example:
    Input: 'I really like New York city'
    Output: 'i really like new your city'
    """
    return text.lower()


def filter_out_uncommon_symbols(text: str) -> str:
    """
    Removes any special character that is not in the
    good symbols list (check regular expression)
    """
    return GOOD_SYMBOLS_RE.sub('', text)


def remove_stopwords(text: str) -> str:
    return ' '.join([x for x in text.split() if x and x not in STOPWORDS])


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


PREPROCESSING_PIPELINE_QA = [
                          replace_special_characters,
                          filter_out_uncommon_symbols,
                          strip_text,
                          remove_redundant_spaces
                          ]

PREPROCESSING_PIPELINE_IR = [
                          lower,
                          replace_special_characters,
                          filter_out_uncommon_symbols,
                          remove_stopwords,
                          strip_text,
                          remove_redundant_spaces
                          ]


# Anchor method
def text_prepare(text: str, ir_mode=False):
    """
    Applies a list of pre-processing functions in sequence (reduce).
    Note that the order is important here!
    """
    filter_methods = PREPROCESSING_PIPELINE_IR if ir_mode else PREPROCESSING_PIPELINE_QA
    return reduce(lambda txt, f: f(txt), filter_methods, text)


# MAIN FUNCTION
def data_preprocessing(orig_df, ir_cleaning=False):
    """
    Clean the data according to the IR flag.
    """

    print('Pre-processing text...')
    print()
    print('[Debug] Before:\n{}'.format(orig_df.context[:3]))
    print()

    input_df = copy.deepcopy(orig_df)
    for label in ['question', 'context']:
        input_df[label] = input_df[label].apply(lambda txt: text_prepare(txt, ir_cleaning))

    print('[Debug] After:\n{}'.format(input_df.context[:3]))
    print()
    print("Pre-processing completed!")

    return input_df

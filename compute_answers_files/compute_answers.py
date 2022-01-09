from settings_file import *
from load_unanswered import load_dataframe
from data_preprocessing import data_preprocessing
from add_data_informations import add_data_informations
from data_conversion import data_conversion

import argparse


OPTS = None


def parse_args():
    parser = argparse.ArgumentParser('Compute Answer script for QA on SQuAD.')
    parser.add_argument('--inputfile', '-in', help='file name in folder /predictions/in/ (es. "unanswered_example.json")', default="unanswered_example.json")
    return parser.parse_args()


def main():
    input_df = load_dataframe(INPUT_FILE_FOLDER + OPTS.inputfile)
    input_df = data_preprocessing(input_df)  # data cleaner
    input_df = add_data_informations(input_df)  # add exact_match & POS
    print(input_df.head)

    x_input_question, x_input_context, x_input_match, x_input_pos_enc = data_conversion(input_df)
    #predict_and_evaluate(input_df, x_input_context, x_input_pos_enc, x_input_match, x_input_question)
    
    return 0


if __name__ == '__main__':
    OPTS = parse_args()
    main()

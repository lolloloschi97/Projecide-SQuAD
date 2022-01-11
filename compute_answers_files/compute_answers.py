from settings_file import *
from load_unanswered import load_dataframe
from data_preprocessing import data_preprocessing
from add_data_informations import add_exact_match, add_POS_tagging
from data_conversion import data_conversion
from model_qa_prediction import predict, load_model
from build_k_input_df import build_k_input_df
from tf_idf_ir import tf_idf_ir
from choose_proposal import choose_best_prediction_proposal

import argparse


OPTS = None


def parse_args():
    parser = argparse.ArgumentParser('Compute Answer script for QA on SQuAD.')
    parser.add_argument('--inputfile', '-in', help='file name in folder /predictions/in/ (es. "unanswered_example.json").', default="unanswered_example.json")
    parser.add_argument('--docretrieval', '-dr', help='Activate document retrieval and select metric. It expects an int as argument: 1 for "cosine" or 2 for "jensenshannon".', type=int, default=0)
    parser.add_argument('--confidenceretrieval', '-cr', help='Expects int [1, 10] (default 5). TOP-K document to retrieve.', type=int, default=5)
    return parser.parse_args()


def check_opts():
    if OPTS.docretrieval not in [0, 1, 2]:
        raise ValueError("Bad argument for --docretrieval: must be 0 - no IR, 1 - IR with cosine, 2 - IR with jensen shannon")
    elif not (1 <= OPTS.confidenceretrieval <= 10 and type(OPTS.confidenceretrieval) == int):
        raise ValueError("Bad argument for --confidenceretrieval: must be an integer in [1, 10].")


def main():
    if not (OPTS.docretrieval == 1 or OPTS.docretrieval == 2):
        #       Only QA enabled
        input_df = load_dataframe(INPUT_FILE_FOLDER + OPTS.inputfile)   # load data
        input_df = data_preprocessing(input_df)  # data cleaner
        input_df = add_exact_match(input_df)  # add exact_match
        input_df = add_POS_tagging(input_df)  # add POS tags
        print("Dataframe loaded and elaborated!")

        x_input_question, x_input_context, x_input_match, x_input_pos_enc = data_conversion(input_df)
        model = load_model()    # load model
        predictions_dict = predict(model, input_df, x_input_context, x_input_pos_enc, x_input_match, x_input_question)
        # save
        cwd = os.getcwd()
        with open(cwd + OUTPUT_FILE_FOLDER + OPTS.inputfile.split('.')[0] + "_answers_no_ir.json", 'w+') as file:
            file.write(json.dumps(dict(predictions_dict)))
    else:
        #       IR + QA pipeline
        complete_input_df = load_dataframe(INPUT_FILE_FOLDER + OPTS.inputfile)  # load data
        input_df_ir = data_preprocessing(complete_input_df, True)  # data cleaner for IR
        best_indexes = tf_idf_ir(input_df_ir, OPTS.docretrieval, OPTS.confidenceretrieval)

        complete_input_df = data_preprocessing(complete_input_df)     # data cleaner for QA
        complete_input_df = add_POS_tagging(complete_input_df)  # add POS tagging
        top_k_input_df = build_k_input_df(complete_input_df, best_indexes, OPTS.confidenceretrieval)  # build K input dataframe based on the computed indexes

        print("Data prepared!")
        model = load_model()    # load model only once
        top_k_predictions_dict = []
        for k, input_df in enumerate(top_k_input_df):
            input_df = add_exact_match(input_df)  # add exact_match now because it depends on the relative query
            x_input_question, x_input_context, x_input_match, x_input_pos_enc = data_conversion(input_df)
            predictions_dict = predict(model, input_df, x_input_context, x_input_pos_enc, x_input_match, x_input_question)
            top_k_predictions_dict.append(predictions_dict)
        # Create a single dictionary containing pairs {query_id: list of k predictions}
        top_k_predictions_dict = {k: [d[k] for d in top_k_predictions_dict] for k in top_k_predictions_dict[0]}
        # Build a single dictionary, similarly to a standard QA case, using tf-idf + jensen shannon distance
        predictions_dict = choose_best_prediction_proposal(top_k_predictions_dict, complete_input_df)
        # save
        cwd = os.getcwd()
        with open(cwd + OUTPUT_FILE_FOLDER + OPTS.inputfile.split('.')[0] + "_answers_ir.json", 'w+') as file:
            file.write(json.dumps(predictions_dict))

    print("RESULTS SAVED in /predictions/out/")
    return 0


if __name__ == '__main__':
    OPTS = parse_args()
    check_opts()
    main()

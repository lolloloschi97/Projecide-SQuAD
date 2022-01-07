from data_loader import data_loader
from data_preprocessing import data_preprocessing
from data_conversion import data_conversion
from model_definition import model_definition
from training import training
from training import load_predict
from compute_start_end_index import compute_start_end_index
from add_data_informations import add_data_informations
from keras.utils.np_utils import to_categorical
from tf_idf_ir import tf_idf_ir

from hyper_param import *


##################################### TODO
#
#                  - fit the tokenizer with both context and question (and text maybe)
#
#
###################################


LOAD_PICKLES = True
TRAINING = True


def save_datasets(train_df, val_df):
    print("Saving datasets...")
    train_df.to_csv(DATASET_ROOT + "training.csv")
    val_df.to_csv(DATASET_ROOT + "validation.csv")
    train_df.to_pickle(DATASET_ROOT + "training_dataframe.pkl")
    val_df.to_pickle(DATASET_ROOT + "validation_dataframe.pkl")
    print("Datasets saved")


def load_datasets():
    print("Loading datasets...")
    train_set = pd.read_pickle(DATASET_ROOT + "training_dataframe.pkl")
    val_set = pd.read_pickle(DATASET_ROOT + "validation_dataframe.pkl")
    print("Datasets loaded")
    return train_set, val_set


# MAIN FUNCTION
def main():
    if LOAD_PICKLES:
        training_df, validation_df = load_datasets()
    else:
        training_df, validation_df = data_loader(TRAIN_SIZE)        # data loader
        training_df, validation_df = data_preprocessing(training_df, validation_df)     # data cleaner
        save_datasets(training_df, validation_df)

    tf_idf_ir(training_df, validation_df)

    quit()
    tokenizer_x, x_train_question, x_train_context, x_train_match, y_train, x_val_question, \
    x_val_context, x_val_match, y_val = data_conversion(training_df, validation_df, LOAD_PICKLES)

    
    # Model
    context_max_lenght = x_train_context.shape[1]
    query_max_lenght = x_train_question.shape[1]
    model = model_definition(context_max_lenght, query_max_lenght, tokenizer_x)

    if TRAINING:
        training(model, x_train_question, x_train_context, x_train_match, y_train, x_val_question, x_val_context, x_val_match, y_val)
    else:
        load_predict(x_val_context, x_val_match, x_val_question, y_val)


if __name__ == '__main__':
    main()

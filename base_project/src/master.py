from data_loader import data_loader
from data_preprocessing import data_preprocessing
from data_conversion import data_conversion
from model_definition import model_definition
from training import training
from training import load_predict
from compute_start_end_index import compute_start_end_index
from hyper_param import *

from keras.utils.np_utils import to_categorical


##################################### TODO
#
#                  - fit the tokenizer with both context and question (and text maybe)
#
#
###################################


LOAD_PICKLES = False   # TRUE for Training, FALSE for TESTING


def save_datasets(train_df, val_df):
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
        training_df, validation_df = compute_start_end_index(training_df, validation_df)     # create the right start and end indexes after cleaning
        save_datasets(training_df, validation_df)

    tokenizer_x, x_train_question, x_train_context, y_train_answer_start, y_train_answer_end, x_val_question,\
              x_val_context, y_val_answer_start, y_val_answer_end = data_conversion(training_df, validation_df, LOAD_PICKLES)

    # one-hot-encoding of y
    y_train_start_enc = to_categorical(y_train_answer_start)
    y_train_end_enc = to_categorical(y_train_answer_end)
    y_val_start_enc = to_categorical(y_val_answer_start)
    y_val_end_enc = to_categorical(y_val_answer_end)
    max_lenght = x_train_context.shape[1]
    y_train_start_enc = np.pad(y_train_start_enc, ((0, 0), (0, max_lenght - y_train_start_enc.shape[1])))
    y_train_end_enc = np.pad(y_train_end_enc, ((0, 0), (0, max_lenght - y_train_end_enc.shape[1])))
    y_val_start_enc = np.pad(y_val_start_enc, ((0, 0), (0, max_lenght - y_val_start_enc.shape[1])))
    y_val_end_enc = np.pad(y_val_end_enc, ((0, 0), (0, max_lenght - y_val_end_enc.shape[1])))

    print(y_train_start_enc.shape)
    print(y_train_end_enc.shape)
    print(y_val_start_enc.shape)
    print(y_val_end_enc.shape)

    # Model
    context_max_lenght = x_train_context.shape[1]
    query_max_lenght = x_train_question.shape[1]
    model = model_definition(context_max_lenght, query_max_lenght, tokenizer_x)

    if not LOAD_PICKLES:
        training(model, x_train_question, x_train_context, y_train_start_enc, y_train_end_enc, x_val_question, x_val_context, y_val_start_enc, y_val_end_enc)
    else:
        load_predict(x_val_question, x_val_context, y_val_start_enc, y_val_end_enc)


if __name__ == '__main__':
    main()
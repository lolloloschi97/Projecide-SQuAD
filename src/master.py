from data_loader import data_loader
from data_preprocessing import data_preprocessing
from data_conversion import data_conversion
from model_definition import model_definition
from training import training
from hyper_param import *

from utils import compute_start_end_index as cmp

LOAD_PICKLES = True


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
        save_datasets(training_df, validation_df)

    cmp.compute_start_end_index(training_df, validation_df)
    quit()

    tokenizer_x, x_train_question, x_train_context, y_train_answer_start, y_train_text, \
    x_val_question, x_val_context, y_val_answer_start, y_val_text = data_conversion(training_df, validation_df, LOAD_PICKLES)

    # Model
    context_max_lenght = x_train_context.shape[1]
    query_max_lenght = x_train_question.shape[1]
    model = model_definition(context_max_lenght, query_max_lenght, tokenizer_x)

    quit()  # FIXME
    training(model, x_train_question, x_train_context, y_train_answer_start, y_train_text, x_val_question, x_val_context, y_val_answer_start, y_val_text)


if __name__ == '__main__':
    main()

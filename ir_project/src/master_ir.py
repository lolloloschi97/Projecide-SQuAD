from data_loader import data_loader
from data_preprocessing import data_preprocessing
from data_conversion import data_conversion
from model_definition import model_definition
from training import training
from training import load_predict
from document_tagging import document_tagging
from query_document_counter import query_document_counter
from hyper_param import *


##################################### TODO
#
#                  - fit the tokenizer with both context and question (and text maybe)
#
#
###################################


LOAD_PICKLES = False   # FALSE for Training, TRUE for TESTING


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
        training_df = training_df[:10000]
        validation_df = validation_df[:10000]
        training_df, validation_df = data_preprocessing(training_df, validation_df)     # data cleaner
        training_df, validation_df = document_tagging(training_df, validation_df)  # pos tagging
        #save_datasets(training_df, validation_df)

    query_document_counter(training_df)
    quit()
    tokenizer_x, x_train_question, x_train_context, y_train, x_val_question, x_val_context, y_val = data_conversion(training_df, validation_df, LOAD_PICKLES)

    # Model
    context_max_lenght = x_train_context.shape[1]
    query_max_lenght = x_train_question.shape[1]
    model = model_definition(context_max_lenght, query_max_lenght, tokenizer_x)

    if not LOAD_PICKLES:
        training(model, x_train_question, x_train_context, y_train, x_val_question, x_val_context, y_val)
    else:
        load_predict(x_val_question, x_val_context, y_val)


if __name__ == '__main__':
    main()

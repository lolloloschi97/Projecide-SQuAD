from data_loader import data_loader
from data_preprocessing import data_preprocessing
from data_conversion import data_conversion
from model_definition import model_definition
from training import training
from training import plot_history
from compute_start_end_index import compute_start_end_index
from add_data_informations import add_data_informations
from prediction import predict_and_evaluate
from hyper_param import *

from keras.utils.np_utils import to_categorical


LOAD_PICKLES = False     # FALSE first time
TRAINING = True
RESUME_TRAINING = False


def save_datasets(train_df, val_df, test_df):
    print("Saving datasets...")
    train_df.to_pickle(DATASET_ROOT + "training_dataframe.pkl")
    val_df.to_pickle(DATASET_ROOT + "validation_dataframe.pkl")
    test_df.to_pickle(DATASET_ROOT + "test_dataframe.pkl")
    train_df.to_csv(DATASET_ROOT + 'training.csv')
    val_df.to_csv(DATASET_ROOT + 'validation.csv')
    test_df.to_csv(DATASET_ROOT + 'test.csv')
    print("Datasets saved")


def load_datasets():
    print("Loading datasets...")
    train_set = pd.read_pickle(DATASET_ROOT + "training_dataframe.pkl")
    val_set = pd.read_pickle(DATASET_ROOT + "validation_dataframe.pkl")
    test_set = pd.read_pickle(DATASET_ROOT + "test_dataframe.pkl")
    print("Datasets loaded")
    return train_set, val_set, test_set


def one_hot_encoding_padding(x_train_pos, y_train_answer_start, y_train_answer_end, x_val_pos, y_val_answer_start, y_val_answer_end,
                             x_test_pos, y_test_answer_start, y_test_answer_end, x_train_context, x_val_context):
    # one-hot-encoding
    print("One hot encoding..")
    x_train_pos_enc = to_categorical(x_train_pos)
    y_train_start_enc = to_categorical(y_train_answer_start)
    y_train_end_enc = to_categorical(y_train_answer_end)
    x_val_pos_enc = to_categorical(x_val_pos)
    y_val_start_enc = to_categorical(y_val_answer_start)
    y_val_end_enc = to_categorical(y_val_answer_end)
    x_test_pos_enc = to_categorical(x_test_pos)
    y_test_start_enc = to_categorical(y_test_answer_start)
    y_test_end_enc = to_categorical(y_test_answer_end)

    # padding
    print("Padding...")
    pos_max_lenght = max(x_train_pos_enc.shape[2], x_val_pos_enc.shape[2])
    x_train_pos_enc = np.pad(x_train_pos_enc, ((0, 0), (0, 0), (0, pos_max_lenght - x_train_pos_enc.shape[2])))
    x_val_pos_enc = np.pad(x_val_pos_enc, ((0, 0), (0, 0), (0, pos_max_lenght - x_val_pos_enc.shape[2])))
    x_test_pos_enc = np.pad(x_test_pos_enc, ((0, 0), (0, 0), (0, pos_max_lenght - x_test_pos_enc.shape[2])))

    max_lenght = max(x_train_context.shape[1], x_val_context.shape[1])
    y_train_start_enc = np.pad(y_train_start_enc, ((0, 0), (0, max_lenght - y_train_start_enc.shape[1])))
    y_train_end_enc = np.pad(y_train_end_enc, ((0, 0), (0, max_lenght - y_train_end_enc.shape[1])))
    y_val_start_enc = np.pad(y_val_start_enc, ((0, 0), (0, max_lenght - y_val_start_enc.shape[1])))
    y_val_end_enc = np.pad(y_val_end_enc, ((0, 0), (0, max_lenght - y_val_end_enc.shape[1])))
    y_test_start_enc = np.pad(y_test_start_enc, ((0, 0), (0, max_lenght - y_test_start_enc.shape[1])))
    y_test_end_enc = np.pad(y_test_end_enc, ((0, 0), (0, max_lenght - y_test_end_enc.shape[1])))

    print(x_train_pos_enc.shape)
    print(x_val_pos_enc.shape)
    print(x_test_pos_enc.shape)
    print(y_train_start_enc.shape)
    print(y_train_end_enc.shape)
    print(y_val_start_enc.shape)
    print(y_val_end_enc.shape)
    print(y_test_start_enc.shape)
    print(y_test_end_enc.shape)

    return x_train_pos_enc, x_val_pos_enc, x_test_pos_enc, y_train_start_enc, y_train_end_enc, y_val_start_enc, y_val_end_enc, y_test_start_enc, y_test_end_enc


# MAIN FUNCTION
def main():
    if LOAD_PICKLES:
        training_df, validation_df, test_df = load_datasets()
    else:
        training_df, validation_df, test_df = data_loader()        # data loader
        training_df, validation_df, test_df = data_preprocessing(training_df, validation_df, test_df)     # data cleaner
        training_df, validation_df, test_df = compute_start_end_index(training_df, validation_df, test_df)     # create the right start and end indexes after cleaning
        training_df, validation_df, test_df = add_data_informations(training_df, validation_df, test_df)     # add exact_match & POS
        save_datasets(training_df, validation_df, test_df)
    print()
    print("Dataset ready!")

    tokenizer_x, x_train_question, x_train_context, x_train_match, x_train_pos, y_train_answer_start, y_train_answer_end, x_val_question,\
              x_val_context, x_val_match, x_val_pos, y_val_answer_start, y_val_answer_end, x_test_question, x_test_context, \
    x_test_match, x_test_pos, y_test_answer_start, y_test_answer_end = data_conversion(training_df, validation_df, test_df, LOAD_PICKLES)

    x_train_pos_enc, x_val_pos_enc, x_test_pos_enc, y_train_start_enc, y_train_end_enc, y_val_start_enc, y_val_end_enc, \
    y_test_start_enc, y_test_end_enc = one_hot_encoding_padding(x_train_pos, y_train_answer_start, y_train_answer_end, x_val_pos, y_val_answer_start,
                             y_val_answer_end, x_test_pos, y_test_answer_start, y_test_answer_end, x_train_context, x_val_context)

    # Model
    context_max_lenght = x_train_context.shape[1]
    query_max_lenght = x_train_question.shape[1]
    model = model_definition(context_max_lenght, query_max_lenght, tokenizer_x, x_train_pos_enc.shape[2])

    if TRAINING:
        training(model, RESUME_TRAINING, x_train_question, x_train_context, x_train_pos_enc, x_train_match, y_train_start_enc, y_train_end_enc, x_val_question, x_val_context, x_val_pos_enc, x_val_match, y_val_start_enc, y_val_end_enc)
    else:
        # plot_history()
        predict_and_evaluate(test_df, x_test_context, x_test_pos_enc, x_test_match, x_test_question, test_df.start_index.to_numpy(), test_df.end_index.to_numpy())


if __name__ == '__main__':
    main()

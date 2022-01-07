from data_loader import data_loader
from data_preprocessing import data_preprocessing
from tf_idf_ir import tf_idf_ir

from hyper_param import *


LOAD_PICKLES = False


def save_datasets(train_df, test_df):
    print("Saving datasets...")
    train_df.to_csv(DATASET_ROOT + "training.csv")
    test_df.to_csv(DATASET_ROOT + "test.csv")
    train_df.to_pickle(DATASET_ROOT + "training_dataframe.pkl")
    test_df.to_pickle(DATASET_ROOT + "test_dataframe.pkl")
    print("Datasets saved")


def load_datasets():
    print("Loading datasets...")
    train_set = pd.read_pickle(DATASET_ROOT + "training_dataframe.pkl")
    test_set = pd.read_pickle(DATASET_ROOT + "test_dataframe.pkl")
    print("Datasets loaded")
    return train_set, test_set


# MAIN FUNCTION
def main():
    if LOAD_PICKLES:
        training_df, test_df = load_datasets()
    else:
        training_df, test_df = data_loader(TRAIN_SIZE)        # data loader
        training_df, test_df = data_preprocessing(training_df, test_df)     # data cleaner
        save_datasets(training_df, test_df)

    tf_idf_ir(training_df, test_df)


if __name__ == '__main__':
    main()

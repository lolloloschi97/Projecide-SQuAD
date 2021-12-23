from data_loader import data_loader
from data_preprocessing import data_preprocessing
from data_conversion import data_conversion
from hyper_param import *


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
        training_df, validation_df = data_loader(TRAIN_SIZE)
        training_df, validation_df = data_preprocessing(training_df, validation_df)
        save_datasets(training_df, validation_df)

    data_conversion(training_df, validation_df, True)



if __name__ == '__main__':
    main()

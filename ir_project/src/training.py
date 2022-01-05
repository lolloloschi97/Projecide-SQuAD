import tensorflow as tf
from hyper_param import *
from matplotlib import pyplot as plt


BATCH_SIZE = 128
EPOCHS = 10


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data, batch_size):
        self.x_context, self.x_context_pos, self.x_context_exact_match, self.x_question = x_data
        self.y = y_data
        self.batch_size = batch_size
        self.num_batches = np.ceil(self.x_context.shape[0] / batch_size)
        self.batch_idx = np.array_split(range(self.x_context.shape[0]), self.num_batches)

    def __len__(self):
        return len(self.batch_idx)

    def __getitem__(self, idx):
        batch_x = (self.x_context[self.batch_idx[idx]], self.x_context_pos[self.batch_idx[idx]], self.x_context_exact_match[self.batch_idx[idx]], self.x_question[self.batch_idx[idx]])
        batch_y = self.y[self.batch_idx[idx]]
        return batch_x, batch_y


def training(model, x_train_question, x_train_context, x_train_context_pos, x_train_context_exact_match, y_train, x_val_question, x_val_context, x_val_context_pos, x_val_context_exact_match, y_val):
    training_generator = DataGenerator((x_train_context, x_train_context_pos, x_train_context_exact_match, x_train_question), (y_train), BATCH_SIZE)
    validation_generator = DataGenerator((x_val_context, x_val_context_pos, x_val_context_exact_match, x_val_question), (y_val), BATCH_SIZE)
    history = model.fit(x=training_generator, epochs=EPOCHS, validation_data=validation_generator)
    model.save(UTILS_ROOT + "saved_model")
    with open(UTILS_ROOT + 'trainHistoryDict', 'wb') as file:
        pickle.dump(history.history, file, pickle.HIGHEST_PROTOCOL)




def load_predict(x_val_context, x_val_context_pos, x_val_context_exact_match, x_val_question, y_val):
    with open(UTILS_ROOT + 'trainHistoryDict', 'rb') as file:
        history_dict = pickle.load(file)
        plotter(history_dict)
    #model = tf.keras.models.load_model(UTILS_ROOT + "saved_model")
    #model.evaluate([x_val_context, x_val_question], [y_val])
    #y_start, y_end = model.predict([x_val_context[:5], x_val_question[:5]])
    #print(y_start.shape)
    #print(y_end.shape)
    #print(np.argmax(y_start, axis=-1))
    #print(np.argmax(y_end, axis=-1))


def plotter(history_dict):
    plt.title('model False Negatives')
    plt.ylabel('false negatives')
    plt.xlabel('epoch')
    plt.plot(history_dict['fls_neg'], 'r')
    plt.plot(history_dict['val_fls_neg'], 'b--')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.plot(history_dict['precision'], 'r')
    plt.plot(history_dict['val_precision'], 'b--')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.plot(history_dict['recall'], 'r')
    plt.plot(history_dict['val_recall'], 'b--')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(history_dict['loss'], 'r')
    plt.plot(history_dict['val_loss'], 'b--')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

import tensorflow as tf
from hyper_param import *
from matplotlib import pyplot as plt


BATCH_SIZE = 64
EPOCHS = 20


def training(model, x_train_question, x_train_context, y_train, x_val_question, x_val_context, y_val):
    history = model.fit(x=[x_train_context, x_train_question], y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([x_val_context, x_val_question], y_val))
    model.save(UTILS_ROOT + "saved_model")
    with open(UTILS_ROOT + 'trainHistoryDict', 'wb') as file:
        pickle.dump(history.history, file, pickle.HIGHEST_PROTOCOL)


def load_predict(x_val_question, x_val_context, y_val):
    with open(UTILS_ROOT + 'trainHistoryDict', 'rb') as file:
        history_dict = pickle.load(file)
        plotter(history_dict)
    model = tf.keras.models.load_model(UTILS_ROOT + "saved_model")
    model.evaluate([x_val_context, x_val_question], [y_val])
    y_start, y_end = model.predict([x_val_context[:5], x_val_question[:5]])
    print(y_start.shape)
    print(y_end.shape)
    print(np.argmax(y_start, axis=-1))
    print(np.argmax(y_end, axis=-1))


def plotter(history_dict):
    plt.title('model binary crossentropy')
    plt.ylabel('crossentropy')
    plt.xlabel('epoch')
    plt.plot(history_dict['bn_cross'], 'r')
    plt.plot(history_dict['val_bn_cross'], 'r--')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.plot(history_dict['precision'], 'r')
    plt.plot(history_dict['val_precision'], 'r--')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.plot(history_dict['recall'], 'r')
    plt.plot(history_dict['val_recall'], 'r--')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(history_dict['loss'], 'r')
    plt.plot(history_dict['val_loss'], 'r--')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

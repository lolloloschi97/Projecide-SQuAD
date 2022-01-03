import tensorflow as tf
from hyper_param import *
from matplotlib import pyplot as plt


BATCH_SIZE = 64
EPOCHS = 20


def training(model, x_train_question, x_train_context, x_train_context_pos, x_train_context_exact_match, y_train_start_enc, y_train_end_enc, x_val_question, x_val_context, x_val_context_pos, x_val_context_exact_match, y_val_start_enc, y_val_end_enc):
    history = model.fit(x=[x_train_context,x_train_context_pos,x_train_context_exact_match, x_train_question], y=[y_train_start_enc, y_train_end_enc], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([x_val_context,x_val_context_pos,x_val_context_exact_match, x_val_question], [y_val_start_enc, y_val_end_enc]))
    model.save(UTILS_ROOT + "saved_model")
    with open(UTILS_ROOT + 'trainHistoryDict', 'wb') as file:
        pickle.dump(history.history, file, pickle.HIGHEST_PROTOCOL)


def load_predict(x_val_question, x_val_context, y_val_start_enc, y_val_end_enc):
    with open(UTILS_ROOT + 'trainHistoryDict', 'rb') as file:
        history_dict = pickle.load(file)
        plotter(history_dict)
    model = tf.keras.models.load_model(UTILS_ROOT + "saved_model")
    model.evaluate([x_val_context, x_val_question], [y_val_start_enc, y_val_end_enc])
    y_start, y_end = model.predict([x_val_context[:5], x_val_question[:5]])
    print(y_start.shape)
    print(y_end.shape)
    print(np.argmax(y_start, axis=-1))
    print(np.argmax(y_end, axis=-1))


def plotter(history_dict):
    plt.title('model binary crossentropy')
    plt.ylabel('crossentropy')
    plt.xlabel('epoch')
    plt.plot(history_dict['tf.nn.softmax_1_bn_cross'], 'r')
    plt.plot(history_dict['tf.nn.softmax_2_bn_cross'], 'b')
    plt.plot(history_dict['val_tf.nn.softmax_1_bn_cross'], 'r--')
    plt.plot(history_dict['val_tf.nn.softmax_2_bn_cross'], 'b--')
    plt.legend(['train_1', 'train_2', 'val_1', 'val_2'], loc='upper left')
    plt.show()

    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.plot(history_dict['tf.nn.softmax_1_precision'], 'r')
    plt.plot(history_dict['tf.nn.softmax_2_precision'], 'b')
    plt.plot(history_dict['val_tf.nn.softmax_1_precision'], 'r--')
    plt.plot(history_dict['val_tf.nn.softmax_2_precision'], 'b--')
    plt.legend(['train_1', 'train_2', 'val_1', 'val_2'], loc='upper left')
    plt.show()

    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.plot(history_dict['tf.nn.softmax_1_recall'], 'r')
    plt.plot(history_dict['tf.nn.softmax_2_recall'], 'b')
    plt.plot(history_dict['val_tf.nn.softmax_1_recall'], 'r--')
    plt.plot(history_dict['val_tf.nn.softmax_2_recall'], 'b--')
    plt.legend(['train_1', 'train_2', 'val_1', 'val_2'], loc='upper left')
    plt.show()

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(history_dict['tf.nn.softmax_1_loss'], 'r')
    plt.plot(history_dict['tf.nn.softmax_2_loss'], 'b')
    plt.plot(history_dict['val_tf.nn.softmax_1_loss'], 'r--')
    plt.plot(history_dict['val_tf.nn.softmax_2_loss'], 'b--')
    plt.legend(['train_1', 'train_2', 'val_1', 'val_2'], loc='upper left')
    plt.show()

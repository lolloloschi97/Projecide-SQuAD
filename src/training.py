import tensorflow as tf
from hyper_param import *


BATCH_SIZE = 64
EPOCHS = 6


def training(model, x_train_question, x_train_context, y_train_start_enc, y_train_end_enc, x_val_question, x_val_context, y_val_start_enc, y_val_end_enc):
    history = model.fit(x=[x_train_context, x_train_question], y=[y_train_start_enc, y_train_end_enc], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([x_val_context, x_val_question], [y_val_start_enc, y_val_end_enc]))
    model.save("saved_model")


def load_predict(x_val_question, x_val_context, y_val_start_enc, y_val_end_enc):
    model = tf.keras.models.load_model("saved_model")
    model.evaluate([x_val_context, x_val_question], [y_val_start_enc, y_val_end_enc])
    y = model.predict([x_val_context[:3], x_val_question[:3]])
    print(np.argmax(y, axis=-1))

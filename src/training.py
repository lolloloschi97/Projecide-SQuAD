
BATCH_SIZE = 64
EPOCHS = 20


def training(model, x_train_question, x_train_context, y_train_answer_start, y_train_text, x_val_question, x_val_context, y_val_answer_start, y_val_text):
    history = model.fit(x=[x_train_context, x_train_question], y=[y_train_answer_start, y_train_text], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([x_val_question, x_val_context], [y_val_answer_start, y_val_text]))

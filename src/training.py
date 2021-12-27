
BATCH_SIZE = 64
EPOCHS = 10


def training(model, x_train_question, x_train_context, y_train_start_enc, y_train_end_enc, x_val_question, x_val_context, y_val_start_enc, y_val_end_enc):
    history = model.fit(x=[x_train_context, x_train_question], y=[y_train_start_enc, y_train_end_enc], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([x_val_question, x_val_context], [y_val_start_enc, y_val_end_enc]))
    model.save("saved_model")

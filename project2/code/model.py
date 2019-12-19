


def generate_model_CNN(vocabulary_size=200, Embedding_size=10, max_length=54):
    model = Sequential()
    model.add(Embedding(vocabulary_size, Embedding_size, input_length=max_length))
    model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.001),metrics=['accuracy'])
    return model


def get_requested_model(model = "CNN"):
    if model == "CNN":
        return generate_model_CNN()
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, SimpleRNN, GRU
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

def embed_train_model(model_savepath, num_classes, embedding_matrix, x_train, y_train, x_test, y_test, vocab_size, maxlen, dim):
    model = Sequential()
    model.add(Embedding(vocab_size, dim, embeddings_initializer= Constant(embedding_matrix), 
                                trainable=False, input_length=maxlen))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics = ['acc'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint(filepath=model_savepath, monitor="val_loss",
                                verbose=1, 
                                save_best_only=True,
                                mode="min")
    callbacks = [checkpoint, early_stop]
    history = model.fit(x_train,
                        y_train,
                        batch_size=64,
                        epochs=30, 
                        validation_split=0.2,
                        callbacks = callbacks,
                        verbose=2)

    return model, history
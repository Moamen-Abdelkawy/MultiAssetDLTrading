import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import os

os.makedirs("../data/models", exist_ok=True)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_save_model(model, X_train, y_train, X_val, y_val, etf_name):
    history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_val, y_val), verbose=1)
    model.save(f"../data/models/{etf_name}_lstm_model.h5")
    with open(f"../data/models/{etf_name}_training_history.pkl", "wb") as file:
        pickle.dump(history.history, file)
    return model, history

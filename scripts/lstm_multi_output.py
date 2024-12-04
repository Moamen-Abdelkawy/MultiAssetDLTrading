from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle

def create_multi_output_lstm(input_shape, output_dim):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_save_multi_output_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                        validation_data=(X_val, y_val), verbose=1)
    model.save("../data/models/multi_output_lstm_model.h5")
    with open("../data/models/multi_output_training_history.pkl", "wb") as file:
        pickle.dump(history.history, file)
    return model, history

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data
import os
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.h5'))
from keras.models import load_model
import keras.losses

# Register 'mse' explicitly
custom_objects = {"mse": keras.losses.mean_squared_error}

model = load_model(model_path, custom_objects=custom_objects)



def predict(stock_symbol):
    data = load_data(stock_symbol)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    last_60_days = scaled_data[-60:]
    X_test = np.array([last_60_days])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0][0]


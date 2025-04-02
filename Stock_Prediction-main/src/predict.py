import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data
import os
import keras.losses

# Load the trained LSTM model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.h5'))
custom_objects = {"mse": keras.losses.mean_squared_error}
model = load_model(model_path, custom_objects=custom_objects)

def predict(stock_symbol):
    # Load the stock data
    data = load_data(stock_symbol)

    # Ensure at least 60 records exist
    if len(data) < 60:
        raise ValueError(f"Not enough data to make a prediction. Found only {len(data)} records.")

    # Normalize only the 'Close' column
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(data[['Close']])  # Using only 'Close' column

    # Extract the last 60 days of the 'Close' price
    last_60_days = scaled_close[-60:]  

    # Ensure shape matches LSTM model expectations
    X_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))  

    # Make prediction
    prediction = model.predict(X_test)

    # Inverse transform the predicted value
    predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))  

    return float(predicted_price[0][0])  # Return as a float value

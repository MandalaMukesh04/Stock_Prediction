import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from data_loader import load_data
from model import create_lstm_model

# Load and preprocess data
stock_data = load_data("AAPL")  # Example: Apple stock
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(stock_data)

# Prepare training data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
X_train, y_train = [], []

for i in range(60, len(train_data)):  # 60 days window
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Train Model
model = create_lstm_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Save Model
model.save("models/lstm_model.h5")

print("Model training completed and saved successfully!")

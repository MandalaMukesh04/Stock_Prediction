import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_loader import load_data
from predict import predict

st.title("📈 Stock Price Prediction using LSTM")

stock = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")

if st.button("Predict"):
    predicted_price = predict(stock)
    st.write(f"### 📊 Predicted Closing Price: ${predicted_price:.2f}")

    # Load & visualize historical data
    data = load_data(stock)

    # 🔍 Debugging: Check type of data
    st.write(f"Data type: {type(data)}")

    if isinstance(data, pd.DataFrame):  # Ensure data is a DataFrame
        st.write("### 🔍 Data Overview")
        st.write(data.head())  # Display first few rows
        st.write(f"Columns: {list(data.columns)}")

        if "Close" in data.columns:
            st.line_chart(data["Close"])
        else:
            st.error("⚠️ Error: 'Close' column not found in dataset!")
    else:
        st.error("⚠️ Error: Data is not a DataFrame. Please check `load_data()` function.")

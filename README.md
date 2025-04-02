# Stock Price Prediction using LSTM

## Overview
This project leverages Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data. The model is trained to forecast stock closing prices using machine learning techniques, making it useful for financial analysis and investment strategies.

## Methodology
The project follows these key steps:

1. **Data Loading**: Fetching historical stock data using Yahoo Finance.
2. **Data Preprocessing**: Normalizing data using MinMaxScaler and structuring it for model training.
3. **Model Training**: Implementing an LSTM neural network to learn patterns from historical stock prices.
4. **Prediction**: Using the trained model to predict future stock prices.
5. **Visualization**: Displaying stock trends and predictions using Streamlit.

## Data Source
The historical stock price data is fetched using the Yahoo Finance API. The dataset consists of closing prices for different stocks, which are then preprocessed for training the LSTM model.

## Tools & Technologies Used
- [**Python**](https://docs.python.org/3/installing/index.html): Core language for data processing and model implementation.
- **TensorFlow/Keras**: For building and training the LSTM model.
- **Scikit-learn**: For data preprocessing and normalization.
- **Pandas & NumPy**: For data manipulation.
- **Matplotlib**: For visualizing stock trends.
- **Streamlit**: For creating a user-friendly web-based interface.
- **Yahoo Finance API**: For fetching real-time and historical stock data.

## Key Features
- **Stock Data Retrieval**: Fetching historical stock data automatically.
- **LSTM-based Prediction**: A deep learning approach for time-series forecasting.
- **User Interface**: A Streamlit-based web app for easy interaction.
- **Real-time Prediction**: Providing insights into potential stock price movements.

## Installation & Setup
### Prerequisites
Ensure you have Python installed. Install the necessary dependencies using:
```sh
pip install pandas numpy tensorflow keras scikit-learn yfinance streamlit matplotlib
```

### Clone the Repository
```sh
git clone https://github.com/your-repo/stock-price-prediction.git
cd stock-price-prediction
```

### Running the Model
1. **Train the Model**
   ```sh
   python train.py
   ```
   This script loads stock data, preprocesses it, trains the LSTM model, and saves it.

2. **Predict Stock Price**
   ```sh
   python predict.py
   ```
   This script loads the trained model and predicts stock prices based on input stock symbols.

3. **Run the Streamlit App**
   ```sh
   streamlit run app.py
   ```
   This launches a web-based interface for stock price prediction.

## Applications
- **Investment Strategy Planning**: Helps investors analyze potential stock movements.
- **Stock Market Research**: Assists in identifying trends and making data-driven decisions.
- **Educational Tool**: Provides insights into LSTM-based forecasting in financial markets.

## Future Enhancements
- **Enhancing Model Accuracy**: Incorporating additional features like trading volume and technical indicators.
- **Multi-stock Analysis**: Predicting multiple stocks simultaneously.
- **Real-time Data Integration**: Fetching and analyzing live stock market data.

## References
- TensorFlow & Keras ([**Documentation**](https://www.tensorflow.org/guide/keras/sequential_model)): For deep learning implementation.
- Yahoo Finance API ([**Documentation**](https://python-yahoofinance.readthedocs.io/en/latest/api.html)): For fetching stock data.
- Scikit-learn ([**Documentation**](https://scikit-learn.org/stable/modules/preprocessing.html)): For data preprocessing and evaluation.


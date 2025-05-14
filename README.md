# Stock_Market_Analysis

This project aims to predict stock prices, specifically IBM stock prices, using deep learning models, including Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and a custom Transformer implementation. The model incorporates time embeddings and moving average techniques for enhanced prediction accuracy.

#Overview
In this project, the goal was to predict the future prices of IBM stock based on historical data using various deep learning models. The following methods were implemented:

Exploratory Data Analysis (EDA): Initial analysis of IBM stock price data to understand trends, patterns, and statistical properties.

RNN & LSTM Models: Recurrent Neural Networks and Long Short-Term Memory models were trained for predicting stock prices.

Time Embeddings & Transformer Model: A custom Transformer was implemented, incorporating time embeddings and a moving average to improve the model's predictive performance.


#Data
The dataset used in this project includes historical stock prices of IBM. It contains the following columns:

Date: The date of the stock price entry.

Open: The opening price of the stock.

High: The highest price during the trading day.

Low: The lowest price during the trading day.

Close: The closing price of the stock.

Volume: The number of shares traded.

The data was sourced from Yahoo Finance or another reliable financial data provider and preprocessed to ensure clean and accurate input to the models.


#Model Architecture
The project uses the following models for stock price prediction:

RNN (Recurrent Neural Network): A basic RNN model was used to capture temporal dependencies in the stock price data.

LSTM (Long Short-Term Memory): An LSTM model, which is more effective at handling long-term dependencies, was trained as an improvement over the basic RNN.

Transformer (Scratch-Implemented): A custom Transformer model was built to learn complex temporal relationships and capture patterns in the stock price data. The Transformer model also integrates time embeddings to enhance its prediction capability.

Moving Average: A moving average was applied to smooth out short-term fluctuations in the stock prices, aiding in trend detection.


#Implementation Details

#Data Preprocessing:
Normalization: The stock prices were normalized to ensure that the model could learn effectively.

Time Series Formatting: The data was reshaped into a time series format suitable for training the models.

#Model Training:
RNN & LSTM: The models were trained using the stock price data, with the output being the predicted price for the next time step.

Transformer: The Transformer model used self-attention mechanisms to understand the temporal relations between past stock prices, further enhanced by incorporating time embeddings.

#Performance Metrics:
The models were evaluated using common performance metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

Moving Average: The moving average model was incorporated as a baseline to evaluate improvements achieved by the Transformer.

#Results
The Transformer-based model with time embeddings outperformed the basic RNN and LSTM models, providing more accurate predictions for IBM stock prices.

The moving average technique helped in reducing noise and improving the model’s generalization.

The inclusion of time embeddings allowed the Transformer model to better capture seasonality and trends in the stock price data.


#installation

# Clone the repository
git clone https://github.com/rohitashbishnoi91/Stock_Market_Analysis.git
cd Stock_Market_Analysis

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# (Optional) Install Jupyter for running notebooks
pip install notebook

# Launch Jupyter Notebook (if using .ipynb files)
jupyter notebook

# OR run the main training script directly (if present)
# python train.py



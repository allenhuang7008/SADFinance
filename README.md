# SADFinance
Sentiment Analysis and Stock Price Movement in Finance 

# Introduction
- The goal of this project is to predict stock prices and explore the impact of sentiment analysis on stock price movements. We combined historical stock data from Yahoo Finance with news data from the Wall Street Journal to gain insights into the relationship between financial news sentiment and stock price fluctuations. In this README file, we provide an overview of the data collection process and the main components of our project.

# Data Collection 
To conduct our analysis, we collected the following key data:

1. Stock Data from Yahoo Finance: We retrieved historical stock data using Yahoo Finance's API. This data spans from the beginning of 2016 to the end of 2018 and includes essential stock information, such as daily open, close, high, and low prices.
2. Stock Symbols: To identify and retrieve data for specific stocks, we needed stock symbols. We acquired a comprehensive list of stock symbols for the S&P 500 companies from https://datahub.io/core/s-and-p-500-companies#data.
3. Wall Street Journal News Data: We obtained valuable news data from the Wall Street Journal through ProQuest TDM Studio. This news data covers the same time period as our stock data and is instrumental in conducting sentiment analysis.

# Sentiment Analysis 
A critical component of our project is the sentiment analysis of the news data. We employed natural language processing techniques to analyze and extract emotion encodings from the Wall Street Journal articles. Emotion encodings derived are "love, anger, disgust, fear, happiness, sadness, surprise, neutral, other". The sentiment analysis process enabled us to capture the sentiment and emotional context embedded in the news articles related to specific stocks.

# LSTM Model for Stock Price Prediction
For stock price prediction, we employed a powerful Long Short-Term Memory (LSTM) model, a type of recurrent neural network (RNN). LSTM models are well-suited for handling sequential data, making them ideal for time series forecasting tasks. By training the LSTM model on historical stock price patterns, we sought to predict whether the stock price would go up or down.

# Integration of Sentiment Analysis Data
A unique aspect of our project lies in the integration of sentiment analysis data into the LSTM model. We explored the potential impact of emotion encodings from news articles on stock price movements. By combining the emotion encodings with historical stock data, our LSTM model endeavors to leverage the additional information from news sentiment analysis to enhance the accuracy of stock price predictions.

# How to run the scripts
- To run **trend prediction**  
`python execute.py true`

- To run **actual price prediction**  
`python execute.py false`

- To adjust **hyperparameter space**  
Change the hyperparameters of objective function in tuning.py. You can also scale up the number of trials and epochs for the tuning process.

- To adjust **training process** after hyperparameter tuning  
The number of epochs per training process is 1000 by default. For the tuning phase, we set n_epochs to 10 for computation efficiency. However, after tuning we would like to perform exhaustive training to achieve better performance. Hence, 1000 epochs and above is preferred.

import numpy as np
import pandas as pd
import pyarrow
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

def pre_stock(path, show_plot=False):
    '''
    This constructs the daily stock market dataframe: {Date, Value, Volume, Price}
    Value = stock_issued * closing_price
    Volume = total issued stock volume on that day
    Price = Value / Volume
    '''
    # retreive stock data
    df = pd.read_parquet(path)

    #compute daily market price mean(stock_issued * stock_price)
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.date
    df['value'] = df.Close * df.Volume
    daily = df.groupby('Date').agg({'value': 'sum', 'Volume': 'sum'})
    daily['price'] = daily.value / daily.Volume
    daily.reset_index(inplace=True)

    if show_plot:
        price = daily.price.values
        plt.plot(daily.index, price)
        plt.xticks(rotation=45)
        plt.show()

    return daily

def time_series_split(data, val_size=0.2, test_size=0.2):
    '''
    data: input can be a numpy array or pandas Series
    val_size and test_size should be in range[0,1]
    '''
    if type(data) != np.ndarray:
        data = np.array(data)
    
    split1 = int(len(data) * (1 - test_size))
    split2 = int(split1 * (1 - val_size))
    train, val, test = np.split(data, [split2, split1])
    
    return train.reshape(-1,1), val.reshape(-1,1), test.reshape(-1,1)

def normalize(train, *arg):
    '''
    This function normalizes the input data, and returns the scaler for inverse transform.
    Note, all data will be normalized using the training data's scaler.
    arg: validation data, test data, etc.
    Return: scaler, norm_train, [norm_arg1, norm_arg2, ...]
    '''
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(train)
    norm_train = scaler.transform(train)
    norm_arg = []
    for data in arg:
        norm_arg.append(scaler.transform(data))
    
    return scaler, norm_train, norm_arg


def create_dataset(data, lookback):
    '''
    This function creates a dataset for time series forecasting, with a rolling window of lookback. 
    The setup of labels depends on the purpose of the model. It can be a period in the future or a single day in the future
    '''
    n_data, n_feat = data.shape
    X = np.empty((n_data-lookback, lookback, n_feat))
    y = np.empty((n_data-lookback, lookback, n_feat))
    for i in range(n_data-lookback):
        feature = data[i:i+lookback]
        target = data[i+1:i+lookback+1]
        X[i] = feature
        y[i] = target
    return torch.tensor(X), torch.tensor(y)


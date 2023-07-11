import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as data
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import sys
import pyarrow
from preprocess import pre_stock, time_series_split, normalize, create_dataset
from LSTM import LSTMModel
import copy

# hyperparameters
lookback = 1
n_epochs = 2000
lr = 0.01
n_nodes = 50
n_layers = 4


def main():
    # retreive stock data
    path = '../../data/stock_data_sp500_2016_2018.parquet'
    daily = pre_stock(path, show_plot=False)

    # split data
    train, val, test = time_series_split(daily.price.values, val_size=0.2, test_size=0.2)

    # normalize data
    scaler, norm_train, (norm_val, norm_test) = normalize(train, val, test)

    # create dataset
    X_train, y_train = create_dataset(norm_train, lookback=lookback)
    X_val, y_val = create_dataset(norm_val, lookback=lookback)
    X_test, y_test = create_dataset(norm_test, lookback=lookback)
    print(f'X_train shape, y_train shape: {X_train.shape}, {y_train.shape}')

    # create dataloader
    batch_size = 10
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=10)

    # setup model
    model = LSTMModel(input_dim=1, n_nodes=n_nodes, output_dim=1, n_layers=n_layers)
    model.double()
    optimizer = opt.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # train model
    min_val_loss = float('inf')

    for t in range(n_epochs):    
        model.train()
        
        for feat, label in loader:
            y_pred = model(feat)
            loss = loss_fn(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        if t % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, y_val).item()

                # Check if current val loss is lower than the minimum val loss
                if val_loss < min_val_loss:
                    # If so, update the minimum val loss and save the current model state
                    min_val_loss = val_loss
                    opt_model_state = copy.deepcopy(model.state_dict())

            print("Epoch %d: train RMSE %.4f, val RMSE %.4f" % (t, np.sqrt(loss.item()), np.sqrt(val_loss)))


    # test model and plot results
    temp_val, _ = create_dataset(scaler.transform(np.concatenate((train[-lookback:], val))), lookback)
    temp_test, test_label = create_dataset(scaler.transform(np.concatenate((val[-lookback:], test))), lookback)
    model.load_state_dict(opt_model_state) # load the optimal model state
    price = daily.price.values
    model.eval()

    with torch.no_grad():
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, 0].view(-1, 1)  # select the 1-D output and reshape from (batch_size, sequence_length) to (batch_size*sequence_length, 1)
        y_pred = scaler.inverse_transform(y_pred)
        train_plot = np.empty_like(price) * np.nan
        train_plot[lookback:len(y_pred)+lookback] = y_pred.flatten()

        val_pred = model(temp_val)
        val_pred = val_pred[:, -1, 0].view(-1, 1)  # select the 1-D output and reshape from (batch_size, sequence_length) to (batch_size*sequence_length, 1)
        val_pred = scaler.inverse_transform(val_pred)
        val_plot = np.empty_like(price) * np.nan
        val_plot[len(y_pred)+lookback:len(y_pred)+len(val_pred)+lookback] = val_pred.flatten()
        
        t_pred = model(temp_test)
        t_pred = t_pred[:, -1, 0].view(-1, 1)  # select the 1-D output and reshape from (batch_size, sequence_length) to (batch_size*sequence_length, 1)
        t_pred = scaler.inverse_transform(t_pred)
        t_plot = np.empty_like(price) * np.nan
        t_plot[len(y_pred)+len(val_pred)+lookback:] = t_pred.flatten()

    plt.figure(figsize=(10,2))
    plt.plot(price, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(val_plot, c='g')
    plt.plot(t_plot, c='orange')
    plt.show()

    t_loss = np.sqrt(mean_squared_error(t_pred, test_label[:,0,0].view(-1, 1)))
    print(f'test RMSE: {t_loss}')
    plt.plot(price[len(y_pred)+len(val_pred)+lookback:], c='b')
    plt.plot(t_plot[len(y_pred)+len(val_pred)+lookback:], c='orange')
    plt.show()


if __name__ == '__main__':
    main()

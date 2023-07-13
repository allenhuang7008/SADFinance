import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import pre_stock, time_series_split, normalize, create_dataset
from LSTM import LSTMModel
from train import train_model
from tuning import tune_model
import json
from sklearn.metrics import mean_squared_error
import torch

def main():
    # retreive data
    stock_path = '../../data/stock_data_sp500_2016_2018.parquet'
    daily = pre_stock(stock_path, show_plot=False)


    # split data
    train, val, test = time_series_split(daily.price.values, val_size=0.2, test_size=0.2)

    # normalize data
    scaler, norm_train, (norm_val, norm_test) = normalize(train, val, test)

    # tune model
    best_params, val_loss = tune_model(norm_train, norm_val)
    print(f'best hyperparameters: {best_params}\nval RMSE: {np.sqrt(val_loss)}')
    with open('../../results/best_params.json', 'w') as f:
        json.dump(best_params, f)

    # train model with best params
    full_train = np.concatenate((norm_train, norm_val))
    min_test_loss, opt_model_state = train_model(best_params, full_train, norm_test)

    # evaluate with test set and plot results
    lookback = best_params['lookback']
    lr = best_params['lr']
    n_nodes = best_params['n_nodes']
    n_layers = best_params['n_layers']
    dropout_rate = best_params['dropout_rate']
    model = LSTMModel(input_dim=1, n_nodes=n_nodes, output_dim=1, n_layers=n_layers, dropout_rate=dropout_rate)
    model.double()

    train = create_dataset(norm_train, lookback=lookback)[0]
    temp_val = create_dataset(scaler.transform(np.concatenate((train[-lookback:], val))), lookback)[0]
    temp_test, test_label = create_dataset(scaler.transform(np.concatenate((val[-lookback:], test))), lookback)
    model.load_state_dict(opt_model_state) # load the optimal model state
    price = daily.price.values
    model.eval()

    with torch.no_grad():
        y_pred = model(train)
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

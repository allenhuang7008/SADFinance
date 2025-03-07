import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import pre_stock, pre_sentiment, time_series_split, normalize, create_dataset
from LSTM import LSTMModel
from train import train_model
from tuning import tune_model
import json
from sklearn.metrics import mean_squared_error
import torch
from sklearn.metrics import roc_curve, auc
import sys

def main(trend, baseline=False):
    # retreive data
    stock_path = '../../data/stock_data_sp500_2016_2018.parquet'
    daily = pre_stock(stock_path, show_plot=False)
    
    if not baseline:
        sentiment_path = '../../data/doc_score_full.csv'
        score = pre_sentiment(sentiment_path)

        # merge data
        daily = daily.merge(score, on='Date', how='left').set_index('Date').to_numpy()
    else:
        daily = daily.set_index('Date').to_numpy()

    # split data
    train, val, test = time_series_split(daily, val_size=0.2, test_size=0.2)

    # normalize data
    scaler, price_scaler, norm_train, (norm_val, norm_test) = normalize(train, val, test)

    # tune model
    best_params, val_loss = tune_model(norm_train, norm_val, trend=trend)
    print(f'best hyperparameters: {best_params}\nval loss: {np.sqrt(val_loss)}')
    if baseline:
        with open('../../results/best_params_baseline.json', 'w') as f:
            json.dump(best_params, f)
    else:
        with open('../../results/best_params_sentiment.json', 'w') as f:
            json.dump(best_params, f)

    # train model with best params
    full_train = np.concatenate((norm_train, norm_val))
    min_test_loss, opt_model_state = train_model(best_params, full_train, norm_test, trend=trend)
    print(f'minimum test BCElogistic: {min_test_loss}')

    # evaluate with test set and plot results
    lookback = best_params['lookback']
    lr = best_params['lr']
    n_nodes = best_params['n_nodes']
    n_layers = best_params['n_layers']
    dropout_rate = best_params['dropout_rate']

    X_train = create_dataset(norm_train, lookback=lookback, trend=trend)[0]
    input_dim = X_train.shape[2]
    temp_val = create_dataset(scaler.transform(np.concatenate((train[-lookback:], val))), lookback, trend=trend)[0]
    temp_test, test_label = create_dataset(scaler.transform(np.concatenate((val[-lookback:], test))), lookback, trend=trend)
    price = daily[:, 0]
    model = LSTMModel(input_dim=input_dim, n_nodes=n_nodes, output_dim=1, n_layers=n_layers, dropout_rate=dropout_rate)
    model.double()
    model.load_state_dict(opt_model_state) # load the optimal model state
    model.eval()

    with torch.no_grad():
        pred = model(X_train)
        pred = pred[:, -1, 0].view(-1, 1)  # select the 1-D output and reshape from (batch_size, sequence_length) to (batch_size*sequence_length, 1)
        y_pred = price_scaler.inverse_transform(pred)
        train_plot = np.empty_like(price) * np.nan
        train_plot[lookback:len(y_pred)+lookback] = y_pred.flatten()

        val_pred = model(temp_val)
        val_pred = val_pred[:, -1, 0].view(-1, 1)  # select the 1-D output and reshape from (batch_size, sequence_length) to (batch_size*sequence_length, 1)
        val_pred = price_scaler.inverse_transform(val_pred)
        val_plot = np.empty_like(price) * np.nan
        val_plot[len(y_pred)+lookback:len(y_pred)+len(val_pred)+lookback] = val_pred.flatten()
        
        t_pred = model(temp_test)        

        if not trend: 
            t_pred = price_scaler.inverse_transform(unscaled_t_pred) # only price need to be rescaled, trend does not
            t_plot = np.empty_like(price) * np.nan
            t_plot[len(y_pred)+len(val_pred)+lookback:] = t_pred.flatten()

            plt.figure(figsize=(10,2))
            plt.plot(price, c='b')
            plt.plot(train_plot, c='r')
            plt.plot(val_plot, c='g')
            plt.plot(t_plot, c='orange')
            plt.savefig('../../results/full_price_pred.png')
            plt.show()

            t_loss = np.sqrt(mean_squared_error(t_pred, test_label[:,0,0].view(-1, 1)))
            print(f'test RMSE: {t_loss}')
            plt.plot(price[len(y_pred)+len(val_pred)+lookback:], c='b')
            plt.plot(t_plot[len(y_pred)+len(val_pred)+lookback:], c='orange')
            plt.savefig('../../results/test_price_pred.png')
            plt.show()

        else:
            unscaled_t_pred = t_pred[:, -1, 0].view(-1, 1)  # select the 1-D output and reshape from (batch_size, sequence_length) to (batch_size*sequence_length, 1)

            # Calculate true positive and false positive rates
            y_true = test_label[:,0,0].view(-1, 1)
            y_score = unscaled_t_pred
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            if baseline:
                plt.savefig('../../results/roc_baseline.png')
            else:
                plt.savefig('../../results/roc_sentiment.png')
            plt.show()

if __name__ == '__main__':
    if sys.argv[1].lower() not in ['true', 'false']:
        raise ValueError('Invalid input: please use True or False')
    trend = sys.argv[1].lower() == 'true'
    main(trend, baseline=True)

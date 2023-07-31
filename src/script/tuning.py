import optuna
import optuna.visualization as vis
from train import train_model

def objective(trial, norm_train, norm_val, trend):
    params = {
        'lookback' : trial.suggest_int("lookback", 5, 60, step=5),
        'lr' : trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        'n_nodes' : trial.suggest_int("n_nodes", 10, 200, step=10),
        'n_layers' : trial.suggest_int("n_layers", 1, 100),
        'dropout_rate' : trial.suggest_float("dropout_rate", 0.2, 0.5)
    }

    val_loss, _ = train_model(params, norm_train, norm_val, trend=trend, n_epochs=100) # here we don't need the optimal state

    return val_loss

def tune_model(norm_train, norm_val, trend, baseline=False):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, norm_train, norm_val, trend=trend), n_trials=1000)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_val_loss = best_trial.value

    # Plot optimization history
    history = vis.plot_optimization_history(study)
    history.show()
    if baseline:
        history.write_image('../../results/optimization_history_baseline.png')
    else:
        history.write_image('../../results/optimization_history_sentiment.png')

    # Plot parameter relationship
    importance = vis.plot_param_importances(study)
    importance.show()
    if baseline:
        importance.write_image('../../results/param_importance_baseline.png')
    else:
        importance.write_image('../../results/param_importance_sentiment.png')
    
    # Plot slice of the parameters
    slice = vis.plot_slice(study, params=['n_layers', 'n_nodes', 'dropout_rate', 'lr'])
    slice.show()
    if baseline:
        slice.write_image('../../results/param_slice_baseline.png')
    else:
        slice.write_image('../../results/param_slice.png')


    return best_params, best_val_loss
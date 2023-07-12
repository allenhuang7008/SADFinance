import optuna
import optuna.visualization as vis
from train import train_model
import matplotlib.pyplot as plt

def objective(trial, norm_train, norm_val):
    params = {
        'lookback' : trial.suggest_int("lookback", 5, 30, step=5),
        'lr' : trial.suggest_loguniform("lr", 1e-5, 1e-1),
        'n_nodes' : trial.suggest_int("n_nodes", 10, 200, step=10),
        'n_layers' : trial.suggest_int("n_layers", 4, 10),
        'dropout_rate' : trial.suggest_uniform("dropout_rate", 0.2, 0.5)
    }

    val_loss, _ = train_model(params, norm_train, norm_val, n_epochs=100) # here we don't need the optimal state

    return val_loss

def tune_model(norm_train, norm_val):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, norm_train, norm_val), n_trials=2)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_val_loss = best_trial.value

    # Plot optimization history
    history = vis.plot_optimization_history(study).show()

    # Plot parameter relationship
    importance = vis.plot_param_importances(study).show()

    # Plot slice of the parameters
    slice = vis.plot_slice(study, params=['n_layers', 'n_nodes', 'dropout_rate', 'lr']).show()

    # save plots in results folder
    plt.savefig('../../results/optimization_history.png')
    plt.savefig('../../results/param_importance.png')
    plt.savefig('../../results/param_slice.png')

    return best_params, best_val_loss
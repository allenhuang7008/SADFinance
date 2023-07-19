# How to run the scripts
- To run trend prediction  
`python execute.py true`

- To run actual price prediction  
`python execute.py false`

- To adjust hyperparameter space  
Change the hyperparameters of objective function in tuning.py. You can also scale up the number of trials and epochs for the tuning process.

- To adjust training process after hyperparameter tuning  
The number of epochs per training process is 1000 by default. For tuning phase we set n_epochs to 10 for computation efficiency. However, after tuning we would like to perform exhaustive training to achieve better performance. Hence, 1000 epochs and above is prefered.

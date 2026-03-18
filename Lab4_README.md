There are two main files relevant to run this code.
Project_Generate_Data.py 
and
ProjectTrain_Hyper.py

Project_Generate_Data.py is needed to generate your own data. It is currently configured to generate 1000 samples with the data_samples variable. You will likely also want to change the folder variable on line 87 to where you would like to save the data. Other than those two variables everything should run on its own.

For ProjectTrain_Hyper.py there are several variables to consider:
TRAIN_FLAG = True  # Set to True to force retraining the model
You will almost always want this to be true unless you are evaluating a saved model on a test set.

USE_TEST = True
This runs a previous model as opposed to the new trained model

USE_OPTUNA = True
Determines whether to train with a hyperparameter search, or only on the parameters given.

NUM_SAMPLES
Number of trials in the hyperparameter search

EPOCHS
Number of EPOCHS, used in both the hyperparameter search and the normal train algorithm.
Note that below 20 epochs the model may not be able to reduce coordinate loss to a point where it can increase the accuracy, resulting in super low accuracy results.

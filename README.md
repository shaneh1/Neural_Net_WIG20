# Neural_Net_WIG20
Neural Network to predict the following days closing price of WIG20
I developed this neural network in an attempt to explore the feasibility of neural networks application in financial prediction.
Reads in normalised data from the csv file (Open, High, Low, Close from day t), returns output of closing price for day t+1.
Uses kersas with tensorflow backend.
Uses softplus activation, adam optimiser and mean squared error.

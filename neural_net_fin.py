import numpy as np
import pandas as pd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

#Read in normalised data and split into train/test/validation samples
data = pd.read_csv("wig20_1.csv", header = None)
data.columns = ["Open", "High", "Low", "Close", "Close_foll"]
len_data = data.shape[0] #Get length of data
train_len = int(round(len_data * 0.8)) #Split it 80/20
test_len = int((round(len_data * 0.2)) + train_len)
training_data = data[:train_len]
test_data = data[train_len:test_len]
#Split into input and output
X_train, Y_train = training_data.drop(["Close_foll"], axis = 1), training_data.Close_foll #Split the input
X_test, Y_test = test_data.drop(["Close_foll"], axis = 1), test_data.Close_foll
#Turn all pandas dataframes into numpy arrays
X_train, Y_train,X_test, Y_test = X_train.values, Y_train.values, X_test.values, Y_test.values
#Transform data into 3d numpy array - required by the NN
train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(train_X.shape, Y_train.shape, test_X.shape, Y_test.shape)


model = Sequential()
model.add(LSTM(8, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.add(Dropout(0.20))
model.add(Activation('softplus'))
model.compile(loss='mse', optimizer='adam')

history = model.fit(train_X, Y_train, validation_split = 0.2, epochs=100, batch_size = 20, validation_data=(test_X, Y_test), verbose=2, shuffle=False)
score = model.evaluate(test_X, Y_test, verbose = False)
print(score)

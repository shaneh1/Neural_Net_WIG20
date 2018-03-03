from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns

#Read in data
data = pd.read_csv('wig20.csv', header = 0)
#Drop Nan columns
data = data.dropna()

#convert dataframe to supervised learning - 4 new columns each 1 step behind originals
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    if type(data) == list:
        n_vars = 1
    else:
        n_vars = data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#Make np array - float
values = data[['Open'] + ['High'] + ['Low'] + ['Close']].values
values = values.astype('float32')

#Normalise features
scaler = MinMaxScaler(feature_range = (0, 1))
scaled = scaler.fit_transform(values)

#Convert to supervise learning
reframed = series_to_supervised(scaled, 1, 1)

#Keep only necessary columns
reframed.drop(reframed.columns[[4, 5, 6]], axis = 1, inplace = True)

#Split data into 70/30 training/test
values = reframed.values
n_train = int(len(values) * 0.7)
train = values[:n_train, :]
test = values[n_train:, :]

#Split into inputs and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

#reshape input to be 3d [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

#Build LSTM - 300 epochs
mdl = Sequential()
mdl.add(LSTM(100, input_shape = (train_X.shape[1], train_X.shape[2])))
mdl.add(Dense(1))
mdl.compile(loss = 'mae', optimizer = 'adamax')
mdl_hist = mdl.fit(train_X, train_y,
                    epochs = 300,
                    batch_size = 100,
                    validation_data = (test_X, test_y),
                    verbose = 0,
                    shuffle = False)

#Make prediction using test_X
yhat = mdl.predict(test_X)

#De-normalise predictions back to original scale
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

#Evaluate performance using RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

#Plot predicted versus actual
pyplot.plot(inv_y, label = 'Actual')
pyplot.plot(inv_yhat, label = 'Predicted')
pyplot.legend(loc = 'best')
pyplot.show()


#import keras modules
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import numpy as np

#create the network
model = Sequential()
model.add(Dense(64,input_dim=3))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('tanh'))

#add optimizer, compile and train
sgd = SGD(lr=0.005)
model.compile(loss='mse', optimizer=sgd)

#load and split dataset
dataset = np.load('xor.npy')
X, y = dataset[:,:2], dataset[:,2]
bias = np.ones((X.shape[0],1))
X = np.concatenate((bias, X), axis=1)
X_train, X_test, y_train, y_test = X[:80000], X[80000:], y[:80000], y[80000:]

#add optimizer, compile and train
sgd = SGD(lr=0.005)
model.compile(loss='mse', optimizer=sgd)
model.fit(X_train, y_train, batch_size=1, epochs=1)

#evaluate loss on test set
loss = model.test_on_batch(X_test, y_test)
print("Loss on the test set :", loss)



# basic imports
import numpy as np
import matplotlib.pyplot as plt
# keras imports
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import losses
from keras import backend as K


def buildModel():
    '''
    This function builds convolutional neural network model.
    '''
    pass
        

if __name__ == "__main__":
    # load data set
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # normalize examples
    X_train = X_train/255.
    X_test = X_test/255.
    # reshape training and test sets
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
    # transform labels
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    
    # implement !
    
    K.clear_session()
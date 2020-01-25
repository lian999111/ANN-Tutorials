# imports
import numpy as np
import matplotlib.pyplot as plt

def metric(y_hat, y):
    y_hat = np.reshape(y_hat, (-1))
    y = np.reshape(y, (-1))
    MSE = np.mean((y_hat - y)**2)

    classifications = y_hat
    classifications[y_hat >= 0] = 1
    classifications[y_hat < 0] = -1

    accuracy = np.sum(classifications == y)/len(y)

    return MSE, accuracy

class MLP():
    '''
    This class implements a simple multilayer perceptron (MLP) with one hidden layer.
    The MLP is trained with true SGD.
    
    | **Args**
    | input_dim:             Number of input units.
    | hidden_layer_units:    Size of hidden layer.
    | eta:                   The learning rate.
    '''
    
    def __init__(self, input_dim=3, hidden_layer_units=10, eta=0.005):
        # input dimensions
        self.input_dim = input_dim
        # number of units in the hidden layer
        self.hidden_layer_units = hidden_layer_units
        # learning rate
        self.eta = eta
        # initialize weights randomly
        # self.weights_input_hidden = np.random.normal(0, 1, (self.hidden_layer_units, input_dim))
        # self.weights_hidden_output = np.random.normal(0, 1, (1, self.hidden_layer_units + 1))
        
        # Xavier initialization
        sigma = 1/(input_dim + self.hidden_layer_units)
        self.weights_input_hidden = np.random.normal(0, sigma**0.5, (self.hidden_layer_units, input_dim))
        sigma = 1/(self.hidden_layer_units + 1)
        self.weights_hidden_output = np.random.normal(0, sigma**0.5, (1, self.hidden_layer_units + 1))
        self.activations = []
    
    def train(self, X, y, epochs=10):
        '''
        Train function of the MLP class.
        This functions trains a MLP using true SGD with a constant learning rate.
        
        | **Args**
        | X:                  Training examples.
        | y:                  Ground truth labels.
        | epochs:             Number of epochs the MLP will be trained.
        '''
        idc = np.arange(X.shape[0])
        shuffled_idc = np.random.shuffle(idc)
        X = X[idc]
        y = y[idc]

        for epoch in range(epochs):
            for x, y_label in zip(X, y):
                # Forward pass
                x = np.reshape(x, (-1, 1))  # make it 2d column vector
                y_hat = self.predict(x, is_training=True)
                loss = (y_hat - y_label)**2

                # h_3 is just y_hat
                h_3 = y_hat
                dLdh_3 = 2 * (y_hat - y_label)     # according to loss function def
                dh_3da_3 = 1 - h_3
                delta_3 = np.multiply(dLdh_3, dh_3da_3)
            
                h_2_with_bias = self.activations.pop()
                h_2 = h_2_with_bias[:-1, :]
                dLdW_2 = delta_3.dot(h_2_with_bias.T)
                
                da_3dh_2 = self.weights_hidden_output[:, 1:].T.dot(delta_3)
                dh_2da_2 = 1 - h_2
                delta_2 = np.multiply(da_3dh_2, dh_2da_2)

                h_1 = x
                dLdW_1 = delta_2.dot(h_1.T)

                self.weights_hidden_output -= self.eta*dLdW_2
                self.weights_input_hidden -= self.eta*dLdW_1

            y_hat = self.predict(X.T)
            MSE, accuracy = metric(y_hat, y)
            print('Epoch: {}, MSE: {}, Accuracy{}'.format(epoch, MSE, accuracy))


    def predict(self, X, is_training=False):
        if is_training:
            self.activations = []

        a_2 = self.weights_input_hidden.dot(X)
        h_2 = np.tanh(a_2)
        # Add bias
        bias = np.ones((1, h_2.shape[1]))
        h_2 = np.concatenate((bias, h_2), axis=0)
        if is_training:
            self.activations.append(h_2)
        
        
        a_3 = self.weights_hidden_output.dot(h_2)
        y_hat = np.tanh(a_3)
        
        return y_hat

if __name__ == "__main__":
    '''
    Train MLPs as defined in the assignments.
    '''
    # load dataset
    dataset = np.load('xor.npy')
    # prepare training data and labels
    X, y = dataset[:,:2], dataset[:,2]
    # add bias neuron
    bias = np.ones((X.shape[0],1))
    X = np.concatenate((bias, X), axis=1)
    # split into training and test
    X_train, X_test, y_train, y_test = X[:80000], X[80000:], y[:80000], y[80000:]

    # To conform to definition in lecture, each layer should be a bunch of column vectors 
    model = MLP(input_dim=3, hidden_layer_units=64, eta=0.005)
    y_hat = model.predict(X_test.T)

    MSE, accuracy = metric(y_hat, y_test)
    print('MSE: {}'.format(MSE))
    print('Accuracy: {}'.format(accuracy))

    y_hat = np.reshape(y_hat, (-1))
    color= ['red' if y == 0 else 'blue' for y in y_hat]
    plt.scatter(X_test[:, 1], X_test[:, 2], color=color, alpha=0.005)
    plt.show()

    model.train(X_train, y_train, epochs=5)
    y_hat = model.predict(X_test.T)
    plt.scatter(X_test[:, 1], X_test[:, 2], color=color, alpha=0.5)
    plt.show()



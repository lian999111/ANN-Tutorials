# imports
import numpy as np
import matplotlib.pyplot as plt

class MLP():
    '''
    This class implements a simple multilayer perceptron (MLP) with one hidden layer.
    The MLP is trained with true SGD.
    
    | **Args**
    | input_dim:             Number of input units.
    | hidden_layer_units:    Size of hidden layer.
    | eta:                   The learning rate.
    '''
    
    def __init__(self, input_dim=2, hidden_layer_units=10, eta=0.005):
        # input dimensions
        self.input_dim = input_dim
        # number of units in the hidden layer
        self.hidden_layer_units = hidden_layer_units
        # learning rate
        self.eta = eta
        # initialize weights randomly
        self.weights_input_hidden = np.random.normal(0, 1, (self.hidden_layer_units, input_dim + 1))
        self.weights_hidden_output = np.random.normal(0, 1, (1, self.hidden_layer_units + 1))
        # activation functions
        self.hidden_act = np.tanh
        self.output_act = np.tanh
        # derivatives of activation functions
        self.hidden_d_act = lambda x: (1 - np.tanh(x)**2)
        self.output_d_act = lambda x: (1 - np.tanh(x)**2)
    
    def train(self, X, y, epochs=100, verbose=0):
        '''
        Train function of the MLP class.
        This functions trains a MLP using true SGD with a constant learning rate.
        
        | **Args**
        | X:                  Training examples.
        | y:                  Ground truth labels.
        | epochs:             Number of epochs the MLP will be trained.
        | verbose:            If non-zero status updates will be printed.
        '''
        for epoch in range(epochs):
            if verbose:
                print('Training epoch ', epoch+1)
            for idx, sample in enumerate(X):
                # forward pass
                # add bias to input
                current_sample = np.concatenate((np.ones((1,1)), np.reshape(sample, (2,1))))
                # hidden layer
                hidden_a = np.dot(self.weights_input_hidden, current_sample)
                hidden_h = self.hidden_act(hidden_a)
                # add bias to hidden layer
                hidden_h = np.concatenate((np.ones((1,1)), hidden_h))
                # output layer
                output_a = np.dot(self.weights_hidden_output, hidden_h)
                output_h = self.output_act(output_a)
                # backward pass
                # deltas at output (dE/dx)
                delta_output = self.output_d_act(output_a) * 2 * (output_h - y[idx])
                # hidden layer deltas
                delta_hidden = self.hidden_d_act(hidden_a) * np.dot(self.weights_hidden_output[:,1:].T, delta_output)
                # update weights
                self.weights_hidden_output -= self.eta * delta_output * hidden_h.T
                self.weights_input_hidden -= self.eta * delta_hidden * current_sample.T
               
    def predict(self, X, y, verbose=0):
        '''
        Test function of the MLP class.
        This function computes the MSE for all examples in X as well as the percentage of misclassifications.
        The classification threshold is considered to be at 0.
        
        | **Args**
        | X:                  Training examples.
        | y:                  Ground truth labels.
        | verbose:            If non-zero MSE and percentage of missclassifications will be printed.
        '''
        mse = 0.
        misclassifications = 0
        misclassified_points = []
        for idx, sample in enumerate(X):
            # forward pass
            current_sample = np.concatenate((np.ones((1,1)), np.reshape(sample, (2,1))))
            # hidden layer
            hidden_h = self.hidden_act(np.dot(self.weights_input_hidden, current_sample))
            # add bias to hidden layer
            hidden_h = np.concatenate((np.ones((1,1)), hidden_h))
            # output layer
            output_h = self.output_act(np.dot(self.weights_hidden_output, hidden_h))
            # compute loss
            mse += (output_h - y[idx])**2
            # classify
            if (output_h >= 0 and y[idx] == 1) or (output_h < 0 and y[idx] == -1):
                pass
            else:
                misclassifications += 1
                misclassified_points += [sample]
        mse /= y.shape[0]
        misclassifications *= 100/y.shape[0]
        if not verbose == 0:
            print('MSE: ', mse)
            print('Misclassifications: ', misclassifications, '%')
        
        return {'mse': mse, 'misclassifications': misclassifications, 'misclassified_points': np.array(misclassified_points)}
    
    def weightsXavierInitialization(self, distribution_type='uniform'):
        '''
        This function implements the weight initialization method introduced by Glorot & Bengio (2010):
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        
        | **Args**
        | distribution_type:  Specifies the distribution type to be used.
        '''
        # this is the "normalized initialization" as described by Glorot & Bengio (2010)
        if distribution_type == 'uniform':
            weight_range = np.sqrt(6)/np.sqrt(self.input_dim + 1 + self.hidden_layer_units)
            self.weights_input_hidden = np.random.uniform(-weight_range, weight_range, (self.weights_input_hidden.shape[0], self.weights_input_hidden.shape[1]))
            weight_range = np.sqrt(6)/np.sqrt(1 + self.hidden_layer_units)
            self.weights_hidden_output = np.random.uniform(-weight_range, weight_range, (self.weights_hidden_output.shape[0], self.weights_hidden_output.shape[1]))
        # initialization as described in equation 12 by Glorot & Bengio (2010)
        elif distribution_type == 'normal':
            weight_std = np.sqrt(2/(self.input_dim + 1 + self.hidden_layer_units))
            self.weights_input_hidden = np.random.normal(0, weight_std, (self.weights_input_hidden.shape[0], self.weights_input_hidden.shape[1]))
            weight_std = np.sqrt(2/(1 + self.hidden_layer_units))
            self.weights_hidden_output = np.random.normal(0, weight_std, (self.weights_hidden_output.shape[0], self.weights_hidden_output.shape[1]))
        
    def predict_class(self, X):
        '''
        Predict function of the MLP class.
        Labels are predicted for all examples in X.
        
        | **Args**
        | X:                  Training examples.
        '''
        predictions = []
        for idx, sample in enumerate(X):
            # forward pass
            current_sample = np.concatenate((np.ones((1,1)), np.reshape(sample, (2,1))))
            # hidden layer
            hidden_h = self.hidden_act(np.dot(self.weights_input_hidden, current_sample))
            # add bias to hidden layer
            hidden_h = np.concatenate((np.ones((1,1)), hidden_h))
            # output layer
            output_h = self.output_act(np.dot(self.weights_hidden_output, hidden_h))
            # store prediction
            predictions += [output_h]
        
        return predictions
    
    def setHiddenLayerActivation(self, activation='tanh'):
        '''
        Sets activation function of the hidden layer units.
        
        | **Args**
        | activation:         Name of the activation function (tanh, relu or arctan).
        '''
        # hyperbolic tangent
        if activation == 'tanh':
            # activation function
            self.hidden_act = np.tanh
            # derivative of activation function
            self.hidden_d_act = lambda x: (1 - np.tanh(x)**2)
        # rectified linear unit
        elif activation == 'relu':
            # activation function
            self.hidden_act = lambda x: np.clip(x, a_min=0, a_max=None)
            # derivative of activation function
            self.hidden_d_act = lambda x: x > 0
        # arctangent
        elif activation == 'arctan':
            # activation function
            self.hidden_act = np.arctan
            # derivative of activation function
            self.hidden_d_act = lambda x: (1/(1+np.arctan(x)**2))

if __name__ == "__main__":
    '''
    Train MLPs as defined in the assignments.
    '''
    # load dataset
    dataset = np.load('xor.npy')
    # prepare training data and labels
    X, y = dataset[:,:2], dataset[:,2]
    # split into training and test
    X_train, X_test, y_train, y_test = X[:80000], X[80000:], y[:80000], y[80000:]
    # define number of hidden layer neurons
    hidden_layer_units = 64
    # learning rate
    eta = 0.005
    # number of runs
    numberOfRuns = 100
    # 2) - use ReLU activation function for hidden layer?
    use_relu = True
    # file identifier
    map_suffix = 'tanh'
    # set relu specific parameters ( Assignment 1.d) )
    if use_relu:
        map_suffix = 'relu'
        hidden_layer_units = 20
    
    # 1.a)
    print('Working on 1.a) ...')
    stats_assignment_01 = []
    # train MLPs
    for run in range(numberOfRuns):
        print('\tstarting run %d' % (run+1))
        mlp = MLP(X.shape[1], hidden_layer_units, eta)
        if use_relu:
            mlp.setHiddenLayerActivation('relu')
        mlp.train(X_train, y_train, 1, verbose=0)
        stats_assignment_01 += [mlp.predict(X_test, y_test)]
        stats_assignment_01[-1]['weights'] = [mlp.weights_input_hidden, mlp.weights_hidden_output]
    # compute stats
    stats_assignment_01_mses = []
    for run in stats_assignment_01:
        stats_assignment_01_mses += [run['mse']]
    stats_assignment_01_mses = np.array(stats_assignment_01_mses)
    stats_assignment_01_mse_mean = np.mean(stats_assignment_01_mses)
    stats_assignment_01_mse_std = np.std(stats_assignment_01_mses, ddof=1)
    # print errors
    print('Assignment 1:\nMean MSE: %f\nSTD: %f' % (stats_assignment_01_mse_mean, stats_assignment_01_mse_std))
    
    # 1.b)
    print('Working on 1.b) ...')
    stats_assignment_02 = []
    # train MLPs
    for run in range(numberOfRuns):
        print('\tstarting run %d' % (run+1))
        mlp = MLP(X.shape[1], hidden_layer_units, eta)
        if use_relu:
            mlp.setHiddenLayerActivation('relu')
        mlp.weightsXavierInitialization()
        mlp.train(X_train, y_train, 1, verbose=0)
        stats_assignment_02 += [mlp.predict(X_test, y_test)]
        stats_assignment_02[-1]['weights'] = [mlp.weights_input_hidden, mlp.weights_hidden_output]
    # compute stats
    stats_assignment_02_mses = []
    for run in stats_assignment_02:
        stats_assignment_02_mses += [run['mse']]
    stats_assignment_02_mses = np.array(stats_assignment_02_mses)
    stats_assignment_02_mse_mean = np.mean(stats_assignment_02_mses)
    stats_assignment_02_mse_std = np.std(stats_assignment_02_mses, ddof=1)
    # print errors
    print('Assignment 1:\nMean MSE: %f\nSTD: %f' % (stats_assignment_02_mse_mean, stats_assignment_02_mse_std))
    
    # 1.b) + c) plot
    plt.figure(2)
    plt.title('Test Error', fontsize=15, position=(0.5, 1.03))
    plt.xlabel('Initialization Method', fontsize=15, position=(0.5, -0.5))
    plt.ylabel('MSE', fontsize=15)
    plt.bar(x=np.array([1, 2]), height=np.array([stats_assignment_01_mse_mean, stats_assignment_02_mse_mean]), width=0.5, tick_label=np.array(['Normal', 'Xavier']), yerr=np.array([stats_assignment_01_mse_std, stats_assignment_02_mse_std]), edgecolor='k')
    plt.savefig('plots/mse_%s.png' % map_suffix, dpi=200, bbox_inches='tight')
    
    # 1.c)
    print('Working on 1.c) ...')
    # compute misclassification heatmaps
    resolution = 100
    assignment_01_heatmap = np.zeros((resolution,resolution))
    assignment_02_heatmap = np.zeros((resolution,resolution))
    for run in range(numberOfRuns):
        for misclassification in stats_assignment_01[run]['misclassified_points']:
            x_coord = min(int(misclassification[0]*resolution), resolution-1)
            y_coord = min(int(misclassification[1]*resolution), resolution-1)
            assignment_01_heatmap[x_coord, y_coord] += 1
        for misclassification in stats_assignment_02[run]['misclassified_points']:
            x_coord = min(int(misclassification[0]*resolution), resolution-1)
            y_coord = min(int(misclassification[1]*resolution), resolution-1)
            assignment_02_heatmap[x_coord, y_coord] += 1
    # rescale heatmaps for better visualization
    assignment_01_heatmap = (np.tanh(assignment_01_heatmap*6/np.amax(assignment_01_heatmap)))
    assignment_02_heatmap = (np.tanh(assignment_02_heatmap*6/np.amax(assignment_02_heatmap)))
    # plot heatmaps
    plt.figure(1, figsize=(10, 5))
    plt.suptitle('Misclassification Maps')
    plt.subplot(121)
    
    plt.axis('equal')
    plt.title('Normally Distributed Weights')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.pcolor(assignment_01_heatmap, vmin=0, cmap='hot')
    plt.colorbar()
    plt.subplot(122)
    plt.axis('equal')
    plt.title('Xavier Initialized Weights')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.pcolor(assignment_02_heatmap, vmin=0, cmap='hot')
    plt.colorbar()
    plt.savefig('plots/misclassified_points_%s.png' % map_suffix, dpi=200)

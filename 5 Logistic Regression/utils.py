import numpy as np 

def gradient(y, x, y_hat):
    # Computes gradient on a given sample
    grad = (y_hat - y) * x
    return grad

def overall_gradient(y, X, y_hat):
    # Computes gradient over a number of samples
    # Input for N samples, M features
    #   y: (N, 1)
    #   X: (N, M+1) M+1 due to bias
    #   y_hat: (N, 1)
    # Output:
    #   overall_gradients: (1, M+1)
    num_samples = y.shape[1]
    grads = gradient(y, X, y_hat)
    return np.sum(grads, axis=1)/num_samples

def logistic_regression(theta, X):
    # Input:
    #   theta: 1-by-M parameter for logistic regression
    #   X:     M-by-N for M features and N samples
    # Output:
    #   y_hat: 1-by-N predicted probabilities of given samples
    return 1 / (1 + np.exp(-theta.dot(X)))

def compute_loss(y, y_hat):
    num_samples = y.shape[1]
    return np.sum(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat)) / num_samples


if __name__ == '__main__':
    # Test gradient() and overall_gradient()
    X = np.reshape(np.arange(1, 13), (4,3))
    y = np.ones((1, 3))
    y_hat = y

    grad = gradient(y[0, 0], X[:, 0], y_hat[0, 0])
    grads = overall_gradient(y, X, y_hat-np.array([[1, 2, 3]]))

    print(grad)
    print(grads)


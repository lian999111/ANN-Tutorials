# %%
import numpy as np 
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import utils

# %%
# Load and preprocess data
data = np.load('05_log_regression_data.npy')
X = data[:, 0:2]
y = data[:, -1]

# Standardize data using zscore
standardized_X = zscore(X)

# Split data
# train_X, test_X are N-by_M now, will be transposed later
train_X, test_X, train_y, test_y = train_test_split(standardized_X, y, test_size=0.2)

# Reshape to match dimension for computation
train_y = np.reshape(train_y, (1, -1))  # an 1-by-N row vector
test_y = np.reshape(test_y, (1, -1))    # an 1-by-N row vector
num_train = train_y.shape[1]
num_test = test_y.shape[1]

# Scatter plot data
fig, ax = plt.subplots(1, 1)
# Use the labels to assign colors, 1: purple, 0: yellow
ax.scatter(train_X[:, 0], train_X[:, 1], c=train_y, alpha=0.5)    
# plt.show()

# %% 
# Gradient descent
epochs = 15000
lr = 0.001

# Initialize theta
theta = np.zeros((1, 3))    # including bias term

# Classifier parameters
p0 = 0.5

# Initialize a list for storing losses
losses = [0]*epochs

# Append train_X with ones for bias and take transpose to match dimension
# extend_train_X is (M+1)-by-N
extend_train_X = np.concatenate((np.ones((num_train,1)), train_X), axis=1).T

for epoch in range(epochs):
    # Forward
    y_hat = utils.logistic_regression(theta, extend_train_X)
    losses[epoch] = utils.compute_loss(train_y, y_hat)

    # Backward
    grads = utils.overall_gradient(train_y, extend_train_X, y_hat)
    theta = theta - lr*grads

# %%
# Inspect training result
y_hat = utils.logistic_regression(theta, extend_train_X)

y_hat[y_hat >= p0] = 1
y_hat[y_hat < p0] = 0

train_acc = np.sum(train_y == y_hat) / num_train
print("Training Accuracy: {}".format(train_acc))

# %%
# Inspect test result
extend_test_X = np.concatenate((np.ones((num_test,1)), test_X), axis=1).T

y_hat = utils.logistic_regression(theta, extend_test_X)

y_hat[y_hat >= p0] = 1
y_hat[y_hat < p0] = 0

test_acc = np.sum(test_y == y_hat) / num_train
print("Test Accuracy: {}".format(test_acc))

# %% 
# Find out the decision boundary
x = np.linspace(-3, 3, num=2)
y = (theta[0, 0] - theta[0, 1]*x)/theta[0, 2]
ax.plot(x, y)
plt.show()

# %%
# Plot the loss vs. epochs
plt.figure()
plt.plot(losses)
plt.xlabel('epochs'), plt.ylabel('loss')
plt.show()

# %%
# Classify using
output = utils.logistic_regression(theta, extend_test_X)
p0_vals = np.arange(0, 1.1, 0.05)
truePosi = np.zeros(p0_vals.shape)
trueNega = np.zeros(p0_vals.shape)
falsePosi = np.zeros(p0_vals.shape)
falseNega = np.zeros(p0_vals.shape)

for idx, p0 in enumerate(p0_vals):
    y_hat = np.zeros((1, num_test))
    y_hat[output >= p0] = 1
    y_hat[output < p0] = 0
    truePosi[idx] = np.sum(np.logical_and(test_y == 1, y_hat == 1))
    trueNega[idx] = np.sum(np.logical_and(test_y == 0, y_hat == 0))
    falsePosi[idx] = np.sum(np.logical_and(test_y == 0, y_hat == 1))
    falseNega[idx] = np.sum(np.logical_and(test_y == 1, y_hat == 0))

precision = truePosi / (truePosi + falsePosi)
recall = truePosi / (truePosi + falseNega)
f1 = 2 / (1/recall + 1/precision)
falsePosiRate = falsePosi / (trueNega + falsePosi)

plt.figure()
plt.plot(recall, precision)
plt.title('Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.figure()
plt.plot(p0_vals, f1)
plt.title('F1 Score')
plt.xlabel('p0')
plt.ylabel('F1')
plt.show()

plt.figure()
plt.plot(falsePosiRate, recall)
plt.title('R.O.C Curve')
plt.xlabel('False Alarm Rate')
plt.ylabel('Recall')
plt.show()

import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score


# Generate 100 samples using f
np.random.seed(1)
num_samples = 100
x = np.linspace(-3, 3, num_samples)
epsilon = np.random.uniform(-0.5, 0.5, 100)
y = utils.f(x) + epsilon

# Plot x-y
plt.figure()
plt.scatter(x, y, label='Data')
plt.title('f(x) = 4x + 5 + epsilon')
plt.xlabel('x'), plt.ylabel('y')

# Linear regression
m = utils.compute_m(x, y)
b = utils.compute_b(x, y, m)
print('m = {0}, b = {1}'.format(m, b))

# Multiple linear regression
X = np.array((np.ones(num_samples), x)).T
Y = np.reshape(y, (-1, 1))
theta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
print(theta_hat)

# Check the fitting result
m_hat = theta_hat[1]
b_hat = theta_hat[0]
y_hat = m_hat * x + b_hat
plt.plot(x, y_hat, color='red', label='Model')
plt.legend()
plt.show()

# Explained variance
ss_data = np.sum((y - np.mean(y))**2)
ss_reg = np.sum((y_hat - np.mean(y))**2)
r_square = ss_reg/ss_data
print('Explained variance r_square: {}'.format(r_square))

# Use scikit-learn to perform linear regression
# help(linear_model.LinearRegression)
reg = linear_model.LinearRegression().fit(np.reshape(x, (-1, 1)), y)
y_hat_sklearn = reg.predict(x)
r_square_sklearn = r2_score(y, y_hat_sklearn)
print('Explained variance r_square from sklearn: {}'.format(y_hat_sklearn))






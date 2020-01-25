import numpy as np
import matplotlib.pyplot as plt

#................Solution 1a................................................#
size = 100
true_params = [5,4]
x   = np.linspace(-3, 3, size)
eps = np.random.uniform(-0.5, 0.5, size)  #noise
Y = true_params[0] + true_params[1]*x + eps

fig, ax = plt.subplots()
ax.scatter(x, Y, c='g', marker='.')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.savefig('1_a.png', format='png')

#................Solution 1b................................................#
m = (np.sum(x*Y) - np.sum(x)*np.sum(Y) / size) / (np.sum(x*x) - 
     ((np.sum(x))**2) / size)
b = np.mean(Y) - m*np.mean(x)
    
print('Estimated intercept is %.4f and slope is %.4f' %(b,m))

#................Solution 1c................................................#
bias = np.ones(size)
X = np.vstack((bias, x)).T     #design matrix

params = np.linalg.pinv((np.dot(X.T,X)))
params = np.dot(params,X.T)
params = np.dot(params,Y)

print('Estimated intercept is %.4f and slope is %.4f' %tuple(params))

#................Solution 1d................................................#
#Plot data with regression line
fig, ax = plt.subplots()

ax.scatter(x, Y, c='g', marker='.')
ax.set_xlabel('x')
ax.set_ylabel('y')

y_pred   = params[0] + params[1] * x 
    
ax.scatter(x, Y, c='g', marker='.', label='data')
ax.plot(x, y_pred, c='r', label='Linear regressor')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('')
ax.legend()
fig.savefig('1_d.png', format='png')

residues = y_pred - Y 

y_mean = np.mean(Y)
ss_data = np.sum((Y - y_mean)**2)
ss_reg= np.sum((y_pred - y_mean)**2)
exp_variance = ss_reg / ss_data
print('Explained variance is = %f' %exp_variance)
#Generate residual plot
fig, ax = plt.subplots()

ax.scatter(x,residues)
ax.plot(x,np.zeros(len(x)), linestyle='-.')
ax.plot(x, 0.5*np.ones(len(x)),c="r", linestyle=':')
ax.plot(x,-0.5*np.ones(len(x)),c="r", linestyle=':')
ax.set_xlabel('x')
ax.set_ylabel('residual')
fig.savefig('1_d2.png', format='png')

#................Solution 1e................................................#

from sklearn import linear_model
from sklearn.metrics import r2_score

lin_reg = linear_model.LinearRegression(fit_intercept=True)
lin_reg.fit(x.reshape(-1,1), Y)
slope = lin_reg.coef_
intercept = lin_reg.intercept_
y_pred_sk = lin_reg.predict(x.reshape(-1,1))
r2_sk = r2_score(Y,y_pred_sk)
print('Estimated intercept is %.4f and slope is %.4f' %(intercept, slope))
print('R2-score (using sklearn) = %f' %r2_sk)

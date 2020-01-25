import numpy as np
import matplotlib.pyplot as plt

#................Solution 3a................................................#
def sgd_update(X,Y,lr=0.001,runs=1):
    ''' X : X matrix like in examples 1 and 2
        Y : observed values of dependent variable
        lr : learning rate
        runs : number of iterations of stochastic gradient descent
    '''
    theta = np.zeros(len(X[0,:]) )                        
    for j in range(runs):        
        p = np.random.permutation(len(X))                
        X = X[p]
        Y = Y[p]        
        for i in range(len(X)) : 
            theta = theta + np.dot(lr*(Y[i] - np.dot(X[i], theta)), X[i]) #eq. 22           
    return theta

#................Solution 3b................................................#
size = 100
true_params = [5,4]
x   = np.linspace(-3, 3, size)
eps = np.random.uniform(-0.5, 0.5, size)  #noise
Y = true_params[0] + true_params[1]*x + eps     
bias = np.ones(100)
X = np.vstack((bias, x)).T 

theta = sgd_update(X, Y, 0.1, 5)

#................Solution 3c................................................#
y_pred   = theta[0] + theta[1] * x 
fig, ax = plt.subplots()
ax.scatter(x, Y, s=8, c='g', marker='.',label='Data')
ax.plot(x, y_pred, c='r',label='Linear regressor')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
# effect of data points 
error_n = []
N = np.arange(5,500,5)
for n in N : 
    x   = np.linspace(-3, 3, n)
    eps = np.random.uniform(-0.5, 0.5, n)  #noise
    Y = true_params[0] + true_params[1]*x + eps 
    bias = np.ones(n)
    X = np.vstack((bias, x)).T 
    params = sgd_update(X,Y,0.01,3)
    
    err = np.linalg.norm(true_params - params, 2)/np.linalg.norm(true_params,2)
    error_n.append(err)

fig, ax = plt.subplots()
ax.plot(N, error_n, linewidth=2, c='g')  
ax.set_xlabel('Data points')
ax.set_ylabel('Error')
# effect of learning rate  
error_lr = []
lr = np.arange(0,0.4,0.001)
 
x   = np.linspace(-3, 3, 100)
eps = np.random.uniform(-0.5, 0.5, 100)  #noise
Y = true_params[0] + true_params[1]*x + eps 
bias = np.ones(100)
X = np.vstack((bias, x)).T 

for l in lr :
    params = sgd_update(X,Y,l,3)    
    err = np.linalg.norm(true_params - params, 2)/np.linalg.norm(true_params,2)
    error_lr.append(err)

fig, ax = plt.subplots()
ax.plot(lr, error_lr, linewidth=2, c='g') 
ax.set_xlabel('Learning rate')
ax.set_ylabel('Error')

    

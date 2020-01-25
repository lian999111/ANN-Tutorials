import numpy as np
import matplotlib.pyplot as plt

#................Solution 2a................................................#
def generate_data(size) : 
    x   = np.linspace(-2,2,size)
    eps = np.random.uniform(-1,1,size)
    true_params = [3,4,-2,1.5,0,-1]
    Y  = (true_params[0] + true_params[1]*x + true_params[2]*(x**2) + 
          true_params[3]*(x**3) + true_params[4]*x**4 + true_params[5]*x**5) + eps          
    bias = np.ones(size)
    X = np.vstack((bias,x,x**2,x**3,x**4,x**5)).T    #design matrix
    
    return x,X,Y
              
size = 100
x, X, Y = generate_data(size) 
fig, ax = plt.subplots()
ax.scatter(x, Y, c='g', marker='.')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.savefig('2_a.png', format='png')

#................Solution 2b................................................#
params = np.linalg.pinv((np.dot(X.T,X)))
params = np.dot(params,X.T)
params = np.dot(params,Y)

fig, ax = plt.subplots()
ax.scatter(x, Y, c='g', marker='.')

y_pred = np.dot(params, X.T)
#OR y_pred = (params[0] + params[1]*x + params[2]*(x**2) + 
#             params[3]*(x**3) + params[4]*x**4 + params[5]*x**5)
ax.scatter(x, Y, c='g', marker='.')
ax.plot(x, y_pred, c='r', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.savefig('2_b.png', format='png')

#................Solution 2c................................................#
error = []
N = np.arange(0,100,5)
true_params = [3,4,-2,1.5,0,-1]
for n in N : 
    x,X,Y = generate_data(n)
    params = np.linalg.pinv((np.dot(X.T,X)))
    params = np.dot(params,X.T)
    params = np.dot(params,Y)    
    err = np.linalg.norm(true_params - params, 2)/np.linalg.norm(true_params,2)
    error.append(err)
    
fig, ax = plt.subplots()
ax.plot(N,error,linewidth=2)
ax.set_xlabel('data points')
ax.set_ylabel('normalized error')
fig.savefig('2_c.png', format='png')

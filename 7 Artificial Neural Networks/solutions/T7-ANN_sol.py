import numpy as np
import matplotlib.pyplot as graph
from scipy.special import expit as sigmoid


def g(x):
    return np.sin(x*2.0*np.pi)

# subtask 1)
def f(x,w,b,v,N):
    output=0.0
    for i in range(N):
        output+=v[i]*sigmoid(w[i]*(x+b[i]))
    return output


# part of subtask 3)
def setParameters(N):
    w=np.ones ((N),dtype=float)*1e4
    b=np.zeros((N),dtype=float)
    v=np.zeros((N),dtype=float)
    
    intervalDomain=np.linspace(0.0,1.0,N/2+1)
    
    for index in range(int(N/2)):
        leftBorder=intervalDomain[index]
        rightBorder=intervalDomain[index+1]
        b[index*2]=-leftBorder
        b[index*2+1]=-rightBorder
        mid=(rightBorder-leftBorder)/2.0+leftBorder
        v[index*2]=g(mid)
        v[index*2+1]=-g(mid)

    return w,b,v


if __name__ == "__main__":
    
    # subtask 2)
    
    x=np.linspace(0.0,1.0,1e4)
    
    # number of units
    N=2
    # vector of weights (hidden)
    w=np.array([1e4,1e4])
    # vector of biases  (hidden)
    b=np.array([-0.4,-0.6])
    # vector of weights (output)
    v=np.array([0.8,-0.8])
    
    # plot the function
    graph.plot(x,f(x,w,b,v,N))
    graph.gca().set_ylim([0.0,1.0])
    
    # display the plot
    graph.show()
    
    
    
    N=4            
    w=np.array([1e4,1e4,1e4,1e4])
    b=np.array([-0.2,-0.4,-0.6,-0.8])
    v=np.array([0.5,-0.5,0.2,-0.2])  
    
    graph.plot(x,f(x,w,b,v,N))
    graph.show()
    
    
    
    N=6                                             
    w=np.array([1e4,1e4,1e4,1e4,1e4,1e4])           
    b=np.array([-0.2,-0.4,-0.4,-0.6,-0.6,-0.8])     
    v=np.array([0.5,-0.5,0.8,-0.8,0.2,-0.2])        
    
    graph.plot(x,f(x,w,b,v,N))
    graph.show()
    

    # subtask 3a)
    
    N=10
    w,b,v=setParameters(N)        
    
    
    graph.plot(x,g(x))
    graph.plot(x,f(x,w,b,v,N))
    graph.show()


    # subtask 3b)
    
    residualError=np.sum(np.abs(f(x,w,b,v,N)-g(x)))
    print('The residual error with %d units is: %f' % (N,residualError))

    errors=np.zeros((10,1),dtype=float)
    N=np.linspace(10,100,10)
    for nIndex in range(N.shape[0]):
        n=int(N[nIndex])
        w,b,v=setParameters(n)
        residualError=np.sum(np.abs(f(x,w,b,v,n)-g(x)))
        errors[nIndex]=residualError

    graph.plot(N,errors)
    graph.show()

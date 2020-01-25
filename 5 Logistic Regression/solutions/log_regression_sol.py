# Load packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# equal aspect ratio  for plots
def set_aspect(ax):
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))        
    
# scatter plot of data points
def plot_data(data):
    fig,ax=plt.subplots(1,1,figsize=(8,8))
    approved=train[train[:,-1]==0] 
    denied=train[train[:,-1]==1]
    ax.plot(approved[:,0],approved[:,1],'go',alpha=0.5)
    ax.plot(denied[:,0],denied[:,1],'ro',alpha=0.5)
    ax.set_xlabel('Residence duration',fontsize=16)
    ax.set_ylabel('Yearly income',fontsize=16)
    return fig,ax

class LogRegression:
    def __init__(self):
        self.loss=[] 
        
    def prepare_data(self,x):
        # add a column for the bias  term 
        x=np.hstack((np.ones((x.shape[0], 1)), x)) 
        return x
    
    def loss_function(self,y,p_hat):
        return -np.mean(y*np.log(p_hat)+(1-y)*np.log(1-p_hat))
    
    def prediction(self,x):
        try: 
            value=np.dot(x,self.theta)
        except:
            # relevant when predictions are done for test set
            x=self.prepare_data(x)
            value=np.dot(x,self.theta)
        return 1/(1+np.exp(-value))
    
    def fit(self,num_epochs,learn_rate,X,Y):
            X=self.prepare_data(X) 
            self.theta=np.zeros(X.shape[1]) # initialize the weights
            for iteration in range(num_epochs):
                p_hat=self.prediction(X)
                gradient = np.dot(X.T, (p_hat-Y))/ Y.size
                self.theta-= learn_rate * gradient # update the weights
                self.loss.append(self.loss_function(Y,p_hat)) 
                
    def plot_decision_boundary(self,ax,X):
        y_intercept=-self.theta[0]/self.theta[2]
        slope=-self.theta[1]/self.theta[2]
        x=np.arange(min(X[:,0]),max(X[:,0]))
        y=slope*x+y_intercept
        ax.plot(x,y,'k-')
        
    def plot_loss(self,num_epochs):
        fig,ax=plt.subplots(1,1)
        ax.plot(range(num_epochs),self.loss)
        ax.set_xlabel('Epochs',fontsize=14)
        ax.set_ylabel('Loss',fontsize=14)
#        fig.savefig('loss.png',dpi=100)

class Classification:
    def __init__(self):
        self.N_thr=8
        self.thresholds=np.linspace(0.0,1,self.N_thr)[::-1] 
        self.precision=[]
        self.recall=[]
        self.F1=[]
        self.FP_rate=[]
        
    def classify(self,true_label,prediction):
        for threshold in self.thresholds:
            response=[1 if item>threshold else 0 for item in prediction]
            accuracy=true_label-response
            FN=len(np.where(accuracy==1)[0])
            FP=len(np.where(accuracy==-1)[0])
            TP=np.sum(true_label)-FN
            TN=len(true_label)-(FN+FP+TP)
            prec=TP/(TP+FP) 
            rec=TP/(TP+FN) 
            
            self.precision.append(prec)
            self.recall.append(rec)
            self.F1.append(2/((1/prec)+(1/rec)))
            self.FP_rate.append(FP/(FP+TN))

    
    def plot_measures(self):
        fig,axes=plt.subplots(1,3,figsize=(8,8))
        colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        [axes[0].plot(self.recall[i],self.precision[i],'o',
        color=colors[i]) for i in range(self.N_thr)]
        axes[0].plot(self.recall,self.precision,'-',alpha=.5)
        axes[0].set_xlabel('Recall',fontsize=16)
        axes[0].set_ylabel('Precision',fontsize=16)        
        
        [axes[1].plot(self.thresholds[i],self.F1[i],'o',
        color=colors[i]) for i in range(self.N_thr)]
        axes[1].plot(self.thresholds,self.F1,'-',alpha=.5)
        axes[1].set_xlabel('Threshold',fontsize=16)
        axes[1].set_ylabel('F1',fontsize=16)
        
        [axes[2].plot(self.FP_rate[i],self.recall[i],'o',
        color=colors[i]) for i in range(self.N_thr)]
        axes[2].plot(self.FP_rate,self.recall,'-',alpha=.5)
        axes[2].plot([0,1],[0,1],'k--')
        axes[2].set_xlabel('False alarms',fontsize=16)
        axes[2].set_ylabel('Recall',fontsize=16)
        axes[2].set_xlim(-0.05,1)
        axes[2].set_ylim(0,1.05)
        [set_aspect(ax) for ax in axes]
        fig.tight_layout()
#        fig.savefig('performance.png',dpi=100)

# Routine for executing the steps
data=np.load('05_log_regression_data.npy')

# standardize the features 
from scipy.stats import zscore 
data[:,0]=zscore(data[:,0])
data[:,1]=zscore(data[:,1])

# split into train and test(validation)
train,test=train_test_split(data)

train_data,train_label=train[:,:-1],train[:,-1]
test_data,test_label=test[:,:-1],test[:,-1]

# visualize the training data
fig,ax=plot_data(train_data) 

reg=LogRegression() # get an instance of the regression model
num_epochs=15000
reg.fit(num_epochs,0.001,train_data,train_label)
reg.plot_decision_boundary(ax,train_data)
#fig.savefig('classification.png',dpi=200)
reg.plot_loss(num_epochs)

# measure the performance of the model on the validation set
test_predictions=reg.prediction(test_data)
decision=Classification()
decision.classify(test_label,test_predictions)
decision.plot_measures()


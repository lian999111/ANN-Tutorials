import numpy as np 
import matplotlib.pyplot as plt

def prepareData(time_steps):
    x = np.linspace(0, 100, time_steps)
    y = np.sin(x) + 0.1 * x
    return x, y

if __name__ == '__main__':
    x, y = prepareData(100)
    plt.plot(x, y)
    plt.show()
import numpy as np

def f(x):
    return 4*x + 5

# For linear regression
def compute_m(x, y):
    num_samples = x.shape[0]
    numerator = np.sum(x*y) - 1/num_samples * np.sum(x) * np.sum(y)
    denominator = np.sum(x**2) - 1/num_samples * np.sum(x)**2
    m = numerator/denominator
    return m

def compute_b(x, y, m):
    num_samples = x.shape[0]
    return 1/num_samples * np.sum(y) - m/num_samples*np.sum(x)

import numpy as np 
from scipy import special
import math

def gen_params(edges, heights):
# Generates parameters of neuron pairs given a series of edges and heights of bars
# Note that length of edges should always be 1 more than length of heights
# A bar can be approimated by a pair of neurons with large weights
    num_bars = len(heights)
    W = 10000 * np.ones((2*num_bars, 1))
    b = np.zeros((2*num_bars, 1))
    for idx, (start, end) in enumerate(zip(edges[0:-1], edges[1:])):
        b[2*idx] = - start * W[2*idx]
        b[2*idx+1] = - end * W[2*idx+1]
    v = np.zeros((2*num_bars, 1))
    for idx, height in enumerate(heights):
        v[2*idx] = height
        v[2*idx+1] = -height
    return W, b, v

def f(W, b, v, x):
    return np.sum(v * special.expit(W*x + b), axis=0)

def g(x):
    return np.sin(2*math.pi*x)
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.array(x > 0, dtype=int)

def ReLU(x):
    return np.maximum(0, x)

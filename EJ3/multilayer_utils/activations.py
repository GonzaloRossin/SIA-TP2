import numpy as np

def sigmoid_activation(H):
    V = 1/(1 + np.exp(-H))
    cache = H
    return V, cache

def sigmoid_backward(dV, cache):
    H = cache
    sigmoid = 1/(1 + np.exp(-H))
    dH = dV * sigmoid * (1 - sigmoid)
    return dH

def relu_activation(H):
    V = max(0.0, H)
    cache = H
    return V, cache

def relu_backward(dV, cache):
    H = cache
    dH = np.array(dV, copy=True)
    dH[H <= 0] = 0
    return dH
import numpy as np

def normalize_sigmoid(Y):
    min = np.min(Y)
    max = np.max(Y)
    Y_range = max - min
    return (1/Y_range) * (Y - min)
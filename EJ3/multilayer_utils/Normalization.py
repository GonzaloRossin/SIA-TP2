import numpy as np

def normalize_sigmoid(Y):
    min_y = np.min(Y)
    max_y = np.max(Y)
    return (Y - min_y) / (max_y - min_y), min_y, max_y

def denormalize_sigmoid(O, min_y, max_y):
    return min_y + O * (max_y - min_y)
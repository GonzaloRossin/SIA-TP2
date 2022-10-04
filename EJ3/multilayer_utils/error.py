import numpy as np
from utils.constants import *

def linear_regression_loss(O, Y):
    D = Y - O
    return np.square(D) / 2

def logistic_regression_loss(O, Y):
    return - np.dot(Y,np.log(O).T) - np.dot(1-Y,np.log(1-O).T)

def compute_error(O, Y, activation):
    m = Y.shape[1]
    if (activation == RELU):
        loss = linear_regression_loss(O, Y)
    else:
        loss = logistic_regression_loss(O,Y)
    error = np.sum(loss) / m
    error = np.squeeze(error)   # fix [[error]] to [error]
    return error
import numpy as np
from utils.constants import *

def linear_regression_loss(O, Y):
    D = Y - O
    return np.square(D) / 2

def logistic_regression_loss(O, Y):
    D = 1-O
    #return - np.dot(Y,np.log(O).T) - np.dot(1-Y,np.log(1-O).T)
    return - np.dot(Y,np.log(O)) - np.dot(1-Y,np.log(D))

def tanh_loss(O, Y):
    loss_pos_1 = np.dot((1+Y).T, np.log(1+Y) - np.log(1+O))
    log_o = np.log(1-O)
    log_y = np.log(1-Y)
    loss_neg_1 = np.dot((1-Y).T, log_y - log_o)
    return loss_pos_1 + loss_neg_1

def compute_error(O, Y, activation):
    m = Y.shape[1]
    if (activation == RELU):
        loss = linear_regression_loss(O, Y)
    elif (activation == SIGMOID):
        loss = logistic_regression_loss(O,Y)
    else:
        loss = tanh_loss(O,Y)
    error = np.sum(loss) / m
    error = np.squeeze(error)   # fix [[error]] to [error]
    return error
import numpy as np
from utils.constants import *
from multilayer_utils.Propagations import model_forward

def logistic_prediction(X, trained_parameters, hidden_activation, apply_bias):
    m = X.shape[1]
    P = np.zeros((1,m))
    O, _ = model_forward(X, trained_parameters, apply_bias, hidden_activation, SIGMOID)
    for i in range(0, O.shape[1]):
        if O[0,i] > 0.5:
            P[0,i] = 1
        else:
            P[0,i] = 0
    return O, P

def linear_prediction(X, trained_parameters, hidden_activation, apply_bias):
    O, _ = model_forward(X, trained_parameters, apply_bias, hidden_activation, RELU)
    return O, _

def tanh_prediction(X, trained_parameters, hidden_activation, apply_bias):
    m = X.shape[1]
    P = np.zeros((1,m))
    O, _ = model_forward(X, trained_parameters, apply_bias, hidden_activation, TANH)
    for i in range(0, O.shape[1]):
        if O[0,i] > 0:
            P[0,i] = 1
        else:
            P[0,i] = -1
    return O, P

def predict(X, trained_parameters, apply_bias, hidden_activation, output_activation):
    if (output_activation == SIGMOID):
        return logistic_prediction(X, trained_parameters, hidden_activation, apply_bias)
    elif (output_activation == RELU):
        return linear_prediction(X, trained_parameters, hidden_activation, apply_bias)
    return tanh_prediction(X, trained_parameters, hidden_activation, apply_bias)
import numpy as np
from utils.constants import *
from multilayer_utils.Propagations import model_forward

def logistic_prediction(X, trained_parameters, apply_bias):
    m = X.shape[1]
    P = np.zeros((1,m))
    probs, _ = model_forward(X, trained_parameters, apply_bias, RELU, SIGMOID)
    for i in range(0, probs.shape[1]):
        if probs[0,i] > 0.5:
            P[0,i] = 1
        else:
            P[0,i] = 0
    return P

def linear_prediction(X, trained_parameters, apply_bias):
    out, _ = model_forward(X, trained_parameters, apply_bias, RELU, RELU)
    return out[-1]

def predict(X, trained_parameters, apply_bias, model_type):
    if (model_type == LOGISTIC):
        return logistic_prediction(X, trained_parameters, apply_bias)
    return linear_prediction(X, trained_parameters, apply_bias)
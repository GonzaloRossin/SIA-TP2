import numpy as np
import perceptron as p
import matplotlib as plt

def accuracy(out_true, out_pred):
    accuracy=np.sum(out_true == out_pred)/len(out_true)
    return accuracy

input=[[-1,1],[1,-1],[-1,-1],[1,1]]

exp_output=[-1,-1,-1,1]

theta, miss_l = p.perceptron(input, exp_output, 0.0001, 100)

print("Accuracy perceptron:", accuracy(exp_output, theta))


"""from data import *
from activation_function import *
import numpy as np

input, expected = data("and")
#input, expected = data("xor")

def perceptron(input, expected, limit, eta, epochs):
    # weight vector
    w = np.zeros(len(input[0]))
    
    # answer vector
    answer = np.ones(len(expected))
    
    # error vector
    error = np.ones(len(expected))
    
    n = 0
    
    # loop for epochs
    while n < epochs:
        for i in range(0, len(input)):
            sum = np.dot(input[i], w)
            
            
            # activation function
            predicted[i] = simple_escalon(sum, limit)
            
            # update weights
            for j in range(0, len(w)):
                w[j] = w[j] + eta*(expected[i] - predicted[i])*input[i][j]
                
            
        n += 1
            
            
            
            
"""



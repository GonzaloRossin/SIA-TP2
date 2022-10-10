import numpy as np
import matplotlib as plt

from data import *
from activation_function import *



def perceptron(input, expected, l_rate, epochs):
    # weight list
    w = np.zeros(len(input[0]))

    # list for predicted values at each epoch
    predicted = np.ones(len(expected))
    
    # empty list to store how many examples were 
    # misclassified at every iteration.
    n_wrong = []
    
    # loop for every epoch
    n = 0
    while n < epochs:
        # store num of misclassified.
        n_miss = 0
        
        # looping for every example.
        for i in range(0,len(input)):
            # mult value for weight
            sum_w_d = np.dot(input[i], w)

            # activation function
            predicted[i] = simple_escalon(sum_w_d, 0)
            
            # checking if prediction is right
            if predicted[i] != expected[i]:
                n_miss += 1

            for j in range(0,len(w)):
            # weight update
                w[j] += l_rate*(expected[i] - predicted[i])*input[i][j]
                
                
        
        # appending number of misclassified examples for each epoch
        n_wrong.append(n_miss)
        n += 1
        
    print("Wrong prediction for last epoch:")
    print(n_wrong)   
     
    return w, n_wrong




            


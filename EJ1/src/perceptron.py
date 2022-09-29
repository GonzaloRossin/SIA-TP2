import numpy as np

import matplotlib as plt

#adicionar bias dps

def perceptron(input,exp_output,l_rate,epochs):
    
    # Initializing parapeters(theta) to zeros.
    # +1 in n+1 for the bias term.
    w = np.zeros(len(input[0]))

    output = np.ones(len(exp_output))
    
    # Empty list to store how many examples were 
    # misclassified at every iteration.
    n_wrong = []
    
    # Training.
    for epoch in range(epochs):
        
        # variable to store #misclassified.
        n_miss = 0
        
        # looping for every example.
        for i in range(0,len(input)):
            
            # Insering 1 for bias, X0 = 1.
            #i = np.insert(i, 0, 1).reshape(-1,1)
            
            # Calculating prediction/hypothesis.
            f= np.dot(input[i], w)

            if f > 0:
                prev=1
            else:
                prev=0

            output[i] = prev

            for j in range(0,len(w)):
            #Perceptron update rule
                w[j] += l_rate*((exp_output[i] - output)*input[i][j])
                
                # Incrementing by 1.
                n_miss += 1
        
        # Appending number of misclassified examples
        # at every iteration.
        n_wrong.append(n_miss)
        
    return w, n_wrong





import numpy as np

# Receives an array containing the dimensions of each layer l
# Returns a dictionary containing W_i and b_i for each layer l
def initialize_parameters(layers_dim, apply_bias=True):
    #np.random.seed(1)
    parameters = {}
    L = len(layers_dim)
    for l in range(1, L):
        parameters['W'+str(l)] = 2 * np.random.rand(layers_dim[l], layers_dim[l-1]) - 1
        if(apply_bias):
            parameters['b'+str(l)] = np.zeros((layers_dim[l], 1))
    return parameters

# Use gradient descent to update parameters
def update_parameters(params, gradients, learning_rate, apply_bias):
    parameters = params.copy()
    if (apply_bias):
        L = len(parameters) // 2
    else:
        L = len(parameters)
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * gradients['dW'+str(l+1)]
        if (apply_bias):
            parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * gradients['db'+str(l+1)]
    return parameters
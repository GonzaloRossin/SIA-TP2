import numpy as np
from multilayer_utils.Error import compute_error
from multilayer_utils.Propagations import model_forward, model_backward
from multilayer_utils.ParametersHandler import initialize_parameters, update_parameters


def multilayer_perceptron(X, Y, input_handler):
    
    np.random.seed(1)
    errors = []
    # TODO: Input layer and output layer size auto detection
    # TODO: Number of hidden units per layer
    layers_dim = [1, 4, 3, 1]
    parameters = initialize_parameters(layers_dim, input_handler.apply_bias)
    
    for i in range(0, input_handler.num_iterations):
        O, caches = model_forward(X, parameters, input_handler.apply_bias, input_handler.hidden_activation, input_handler.output_activation)
        error = compute_error(O, Y, input_handler.output_activation)
        errors.append(error)
        gradients = model_backward(O, Y, caches, input_handler.hidden_activation, input_handler.output_activation, input_handler.apply_bias)
        parameters = update_parameters(parameters, gradients, input_handler.learning_rate, input_handler.apply_bias)
    
    return parameters, errors
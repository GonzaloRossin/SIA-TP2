import json
import numpy as np
from utils.input_handler import InputHandler
from multilayer_utils.Predictions import predict
from multilayer_utils.Normalization import normalize_sigmoid
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron

def main():

    with open('config.json', 'r') as f:
        json_file = json.load(f)
        input_handler = InputHandler(json_file)

    # TODO: Process input files
    train_X = np.array([[-1,1,-1,1],[1,-1,-1,1]])
    train_Y = np.array([[1,1,-1,-1]])

    print(f"Train Y shape is {train_Y.shape}")
    print(f"Train X shape is {train_X.shape}")
    
    print(train_Y)

    normalized_Y = normalize_sigmoid(train_Y)

    print(normalized_Y)
    
    '''
    test_X, test_Y = []
    '''
    
    # Training
    parameters, errors = multilayer_perceptron(train_X, normalized_Y, input_handler)
    #print(f"These are the parameters\n {parameters}")
    #print(f"\nThese are the errors\n {errors}")

    # Predictions
    predictions_training = predict(train_X, parameters, input_handler.apply_bias, input_handler.model_type)
    print(f"\nThese are the predictions\n {predictions_training}")

    #predictions_test = predict(test_X, test_Y, input_handler.apply_bias, input_handler.model_type)

    # TODO: Graphics


if __name__ == "__main__":
    main()
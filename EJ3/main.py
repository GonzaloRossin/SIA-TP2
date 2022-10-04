import json
from utils.input_handler import InputHandler
from multilayer_utils.Predictions import predict
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron

def main():

    with open('config.json', 'r') as f:
        json_file = json.load(f)
        input_handler = InputHandler(json_file)

    # TODO: Process input files
    train_X, train_Y = []
    test_X, test_Y = []
    
    # Training
    parameters, errors = multilayer_perceptron(train_X, train_Y, input_handler)
    
    # Predictions
    predictions_training = predict(train_X, train_Y, input_handler.apply_bias, input_handler.model_type)
    predictions_test = predict(test_X, test_Y, input_handler.apply_bias, input_handler.model_type)

    # TODO: Graphics
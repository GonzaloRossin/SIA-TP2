import json
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import LOGISTIC
from utils.InputHandler import InputHandler
from multilayer_utils.Predictions import predict
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron
from multilayer_utils.Normalization import denormalize

def ejA_main():

    with open('config.json', 'r') as f:
        json_file = json.load(f)
        input_handler = InputHandler(json_file)

    train_X = input_handler.training_set_X
    train_Y = input_handler.training_set_Y
    
    '''
    print(f"train_X = \n{train_X}\n")
    print(f"train_Y = {train_Y}\n")
    print(f"test_X = \n{test_X}\n")
    print(f"test_Y = {test_Y}\n")
    '''
    
    # Training
    parameters, errors = multilayer_perceptron(train_X, train_Y, input_handler)

    # Predictions
    if (input_handler.ratio == 100):    # if training set uses all the dataset
        test_X = train_X
        test_Y = train_Y
    else:    
        test_X = input_handler.test_set_X
        test_Y = input_handler.test_set_Y
    
    O, _ = predict(test_X, parameters, input_handler.apply_bias, input_handler.model_type)

    '''
    print(f"Y_norm = {np.squeeze(train_Y)}\n")
    print(f"Prediction = {np.squeeze(P)}\n")  # 0/1 Predictions
    '''
    
    if (input_handler.normalize):
        print(f"Expected = {np.squeeze(denormalize(test_Y, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
        print(f"Denormalized Output = {np.squeeze(denormalize(O, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
    else:
        print(f"Expected = {np.squeeze(test_Y)}\n")
        print(f"Output = {np.squeeze(O)}\n")

    '''
    print(f"Trained parameters\n {parameters}\n")
    '''

    # Graphics
    fig = plt.subplot()
    fig.set_title("Error function")
    fig.set_xlabel("Epochs")
    fig.set_ylabel("Error")
    #fig.set_ylim([0, 1])
    #fig.plot(errors, marker='o')
    fig.plot(errors)
    plt.show()

    # TODO: Output graphics


if __name__ == "__main__":
    ejA_main()
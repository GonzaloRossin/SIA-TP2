import json
import numpy as np
import matplotlib.pyplot as plt
from utils.InputHandler import InputHandler
from multilayer_utils.Normalization import denormalize
from multilayer_utils.Prediction import predict_multiclass
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron

def ejC_main():

    with open('config.json', 'r') as f:
        json_file = json.load(f)
        input_handler = InputHandler(json_file)

    train_X = input_handler.training_set_X
    train_Y = input_handler.training_set_Y
    
    # Training
    parameters, errors = multilayer_perceptron(train_X, train_Y, input_handler)

    # Predictions
    if (input_handler.ratio == 100):    # if training set uses all the dataset
        test_X = train_X
        test_Y = train_Y
    else:    
        test_X = input_handler.test_set_X
        test_Y = input_handler.test_set_Y
    
    O, P = predict_multiclass(test_X, parameters, input_handler.hidden_activation, input_handler.output_activation, input_handler.apply_bias)
    
    '''
    print(f"Y_norm = {np.squeeze(train_Y)}\n")
    print(f"Trained parameters\n {parameters}\n")
    '''
    
    if (input_handler.normalize):
        print(f"Expected = {np.squeeze(denormalize(test_Y, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
        print(f"Denormalized Output = {np.squeeze(denormalize(O, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
        denormalized_P = denormalize(P, input_handler.min_y, input_handler.max_y, input_handler.output_activation)
        print(f"Prediction =\n{np.squeeze(denormalized_P)}\n")  # 0/1 Predictions
        print(f"Accuracy =  {np.mean((denormalized_P == test_Y)) * 100}%\n")
    else:
        print(f"Expected =\n{np.squeeze(test_Y)}\n")
        print(f"Output = {np.squeeze(O)}\n")
        print(f"Prediction =\n{np.squeeze(P)}\n")  # 0/1 Predictions
        print(f"Accuracy =  {np.mean((P == test_Y)) * 100}%\n")

    # Graphics
    fig = plt.subplot()
    fig.set_title("Error function")
    fig.set_xlabel("Epochs")
    fig.set_ylabel("Error")
    fig.plot(errors)
    plt.show()


if __name__ == "__main__":
    ejC_main()
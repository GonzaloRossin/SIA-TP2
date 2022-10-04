import json
import numpy as np
import matplotlib.pyplot as plt
from utils.input_handler import InputHandler
from multilayer_utils.Predictions import predict
from multilayer_utils.Normalization import normalize_sigmoid
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron

def ejA_main():

    train_X = np.array([[-1,1,-1,1],[1,-1,-1,1]])
    train_Y = np.array([[1,1,-1,-1]])
    
    print(f"\nX =\n{train_X}\n")
    print(f"Y = {np.squeeze(train_Y)}")

    with open('config.json', 'r') as f:
        json_file = json.load(f)
        input_handler = InputHandler(json_file)

    normalized_Y = normalize_sigmoid(train_Y)
    print(f"Y_norm = {np.squeeze(normalized_Y)}\n")
    
    # Training
    parameters, errors = multilayer_perceptron(train_X, normalized_Y, input_handler)
    #print(f"Trained parameters\n {parameters}")

    # Predictions
    O, P = predict(train_X, parameters, input_handler.apply_bias, input_handler.model_type)
    print(f"Output = {np.squeeze(O)}\n")  # Direct output
    print(f"Prediction = {np.squeeze(P)}\n")  # 0/1 Predictions

    # Graphics
    fig = plt.subplot()
    fig.set_title("Error function")
    fig.set_ylabel("Error")
    fig.set_xlabel("Epochs")
    fig.plot(errors)
    plt.show()

    # TODO: Output graphics


if __name__ == "__main__":
    ejA_main()
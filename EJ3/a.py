import json
import numpy as np
import matplotlib.pyplot as plt
from utils.plotter import plot_decision_boundary
from utils.constants import LOGISTIC
from utils.InputHandler import InputHandler
from multilayer_utils.Prediction import predict, predict_decision_boundary
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron
from multilayer_utils.Normalization import denormalize

def ejA_main():

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
    
    O, P = predict(test_X, parameters, input_handler.apply_bias, input_handler.hidden_activation, input_handler.output_activation)

    '''
    print(f"Y_norm = {np.squeeze(train_Y)}\n")
    '''
    
    if (input_handler.normalize):
        P = denormalize(P, input_handler.min_y, input_handler.max_y, input_handler.output_activation)
        test_Y = np.squeeze(denormalize(test_Y, input_handler.min_y, input_handler.max_y, input_handler.output_activation))
        print(f"\nDenormalized Output = {np.squeeze(denormalize(O, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
    else:
        test_Y = np.squeeze(test_Y)
        print(f"Output = {np.squeeze(O)}\n")

    print(f"Prediction = {np.squeeze(P)}\n")  # 0/1 Predictions
    print(f"Expected = {test_Y}\n")
    print(f"Accuracy =  {np.mean((P == test_Y)) * 100}%\n")

    '''
    print(f"Trained parameters\n {parameters}\n")
    '''

    # Graphics
    fig, axs = plt.subplots(1,2)
    axs[0].set_title('Data classification')
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")
    plot_decision_boundary(lambda x: predict_decision_boundary(x.T,parameters,input_handler.apply_bias,input_handler.hidden_activation), test_X, axs[0])
    pairs_X = test_X.T
    axs[0].scatter(pairs_X[:,0], pairs_X[:,1], s=100, c=test_Y)
    axs[1].set_title("Error function")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Error")
    #fig.set_ylim([0, 1])
    axs[1].plot(errors)
    
    plt.show()


if __name__ == "__main__":
    ejA_main()
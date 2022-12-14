import json
import numpy as np
import matplotlib.pyplot as plt
from utils.InputHandler import InputHandler
from multilayer_utils.Prediction import predict
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron
from multilayer_utils.Normalization import denormalize

def ejB_ratios():

    # 1 es PAR, 0 es IMPAR

    errors = []
    accs = []
    training_set_ratios = [100, 50, 25]

    for ratio in training_set_ratios:

        acc = []

        for i in range(5):

            with open('config.json', 'r') as f:
                json_file = json.load(f)
                input_handler = InputHandler(json_file, ratio)

            train_X = input_handler.training_set_X
            train_Y = input_handler.training_set_Y
                
            # Training
            parameters, error = multilayer_perceptron(train_X, train_Y, input_handler)
            errors.append(error)

            # Predictions
            if (input_handler.ratio == 100):    # if training set uses all the dataset
                test_X = train_X
                test_Y = train_Y
            else:    
                test_X = input_handler.test_set_X
                test_Y = input_handler.test_set_Y
            
            O, P = predict(test_X, parameters, input_handler.apply_bias, input_handler.hidden_activation, input_handler.output_activation)

            print(f"Y_norm = {np.squeeze(train_Y)}\n")
            
            if (input_handler.normalize):
                print(f"Expected = {np.squeeze(denormalize(test_Y, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
                print(f"Denormalized Output = {np.squeeze(denormalize(O, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
                denormalized_P = denormalize(P, input_handler.min_y, input_handler.max_y, input_handler.output_activation)
                print(f"Prediction = {np.squeeze(denormalized_P)}\n")  # 0/1 Predictions
                print(f"Accuracy =  {np.mean((denormalized_P == test_Y)) * 100}%\n")
                acc.append(np.mean((denormalized_P == test_Y)) * 100)
            else:
                print(f"Expected = {np.squeeze(test_Y)}\n")
                print(f"Output = {np.squeeze(O)}\n")
                print(f"Prediction = {np.squeeze(P)}\n")  # 0/1 Predictions
                print(f"Accuracy =  {np.mean((P == test_Y)) * 100}%\n")
                acc.append(np.mean((P == test_Y)) * 100)

            print(f"Trained parameters\n {parameters}\n")
        
        '''
        for err in errors:
            fig = plt.subplot()
            fig.set_title("Error function")
            fig.set_xlabel("Epochs")
            fig.set_ylabel("Error")
            fig.plot(err)
            plt.show()
        '''

        accs.append(np.mean(acc))

    percents = list(map(lambda p: str(p)+"%",training_set_ratios))
    fig = plt.figure(figsize=(10,5))
    bars = plt.bar(percents, accs, width=0.4)
    plt.title("Accuracy by ratio")
    plt.xlabel("Ratio")
    plt.ylabel("Accuracy")
    plt.bar_label(bars)
    plt.show()

def ejB_main():

    # 1 es PAR, 0 es IMPAR

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
        print(f"Expected = {np.squeeze(denormalize(test_Y, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
        print(f"Denormalized Output = {np.squeeze(denormalize(O, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
        denormalized_P = denormalize(P, input_handler.min_y, input_handler.max_y, input_handler.output_activation)
        print(f"Prediction = {np.squeeze(denormalized_P)}\n")  # 0/1 Predictions
        print(f"Accuracy =  {np.mean((denormalized_P == test_Y)) * 100}%\n")
    else:
        print(f"Expected = {np.squeeze(test_Y)}\n")
        print(f"Output = {np.squeeze(O)}\n")
        print(f"Prediction = {np.squeeze(P)}\n")  # 0/1 Predictions
        print(f"Accuracy =  {np.mean((P == test_Y)) * 100}%\n")

    '''
    print(f"Trained parameters\n {parameters}\n")
    '''

    fig = plt.subplot()
    fig.set_title("Error function")
    fig.set_xlabel("Epochs")
    fig.set_ylabel("Error")
    fig.plot(errors)
    plt.show()


if __name__ == "__main__":
    ejB_main()
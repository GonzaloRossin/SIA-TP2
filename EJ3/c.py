import json
import numpy as np
import matplotlib.pyplot as plt
from utils.InputHandler import InputHandler
from multilayer_utils.Normalization import denormalize
from multilayer_utils.Prediction import predict_multiclass
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron

def ejC_optimizer():

    errors = []
    accs = []
    optimizers = ["grad desc","momentum", "adam"]

    for optimizer in optimizers:

        acc = []

        for i in range(5):

            with open('config.json', 'r') as f:
                json_file = json.load(f)
                input_handler = InputHandler(json_file)

            train_X = input_handler.training_set_X
            train_Y = input_handler.training_set_Y

            input_handler.optimizer = optimizer
                
            # Training
            parameters, error = multilayer_perceptron(train_X, train_Y, input_handler)

            # Predictions
            if (input_handler.ratio == 100):    # if training set uses all the dataset
                test_X = train_X
                test_Y = train_Y
            else:    
                test_X = input_handler.test_set_X
                test_Y = input_handler.test_set_Y
            
            O, P = predict_multiclass(test_X, parameters, input_handler.hidden_activation, input_handler.output_activation, input_handler.apply_bias)

            print(f"Y_norm = {np.squeeze(train_Y)}\n")
            
            if (input_handler.normalize):
                print(f"Expected = {np.squeeze(denormalize(test_Y, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
                print(f"Denormalized Output = {np.squeeze(denormalize(O, input_handler.min_y, input_handler.max_y, input_handler.output_activation))}\n")
                denormalized_P = denormalize(P, input_handler.min_y, input_handler.max_y, input_handler.output_activation)
                print(f"Prediction =\n{np.squeeze(denormalized_P)}\n")  # 0/1 Predictions
                print(f"Accuracy =  {np.mean((denormalized_P == test_Y)) * 100}%\n")
                acc.append(np.mean((denormalized_P == test_Y)) * 100)
            else:
                print(f"Expected =\n{np.squeeze(test_Y)}\n")
                print(f"Output = {np.squeeze(O)}\n")
                print(f"Prediction =\n{np.squeeze(P)}\n")  # 0/1 Predictions
                print(f"Accuracy =  {np.mean((P == test_Y)) * 100}%\n")
                acc.append(np.mean((P == test_Y)) * 100)

            #print(f"Trained parameters\n {parameters}\n")

        accs.append(np.mean(acc))
        errors.append(error)

    #'''
    fig = plt.subplot()
    fig.set_title("Error function")
    fig.set_xlabel("Epochs")
    fig.set_ylabel("Error")
    for err in errors:
        fig.plot(err)
    fig.legend(optimizers)
    plt.show()
    #'''

    fig = plt.figure(figsize=(10,5))
    bars = plt.bar(optimizers, accs, width=0.4)
    plt.title("Accuracy by optimizer")
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy")
    plt.bar_label(bars)
    plt.show()


def ejC_etha():

    with open('config.json', 'r') as f:
        json_file = json.load(f)
        input_handler = InputHandler(json_file)

    train_X = input_handler.training_set_X
    train_Y = input_handler.training_set_Y
    
    ethas = [0.1, 0.0001, 0.000001]
    errors = {}

    for etha in ethas:

        input_handler.learning_rate = etha

        # Training
        parameters, error = multilayer_perceptron(train_X, train_Y, input_handler)
        errors[etha] = error

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
    ordered_ethas = errors.keys()
    for etha in ordered_ethas:
        fig.plot(errors[etha])
    fig.legend(ordered_ethas)
    plt.show()

def ejC_noise():

    np.random.seed(1)

    with open('config.json', 'r') as f:
        json_file = json.load(f)
        input_handler = InputHandler(json_file)

    train_X = input_handler.training_set_X
    train_Y = input_handler.training_set_Y
    
    # Training
    parameters, errors = multilayer_perceptron(train_X, train_Y, input_handler)

    # Predictions over training set
    if (input_handler.ratio == 100):
        test_X = train_X
        test_Y = train_Y

    accs = []

    acc = []
    for i in range(5):
        O, P = predict_multiclass(test_X, parameters, input_handler.hidden_activation, input_handler.output_activation, input_handler.apply_bias)
        '''
        print(f"Expected =\n{np.squeeze(test_Y)}\n")
        print(f"Output = {np.squeeze(O)}\n")
        print(f"Prediction =\n{np.squeeze(P)}\n")  # 0/1 Predictions
        print(f"Accuracy =  {np.mean((P == test_Y)) * 100}%\n")
        '''    
        acc.append(np.mean((P == test_Y)) * 100)
    accs.append(np.mean(acc))

    noise1 = np.random.rand(test_X.shape[0],test_X.shape[1])
    test_X_noise_1 = test_X + noise1

    acc = []
    for i in range(5):
        O, P = predict_multiclass(test_X_noise_1, parameters, input_handler.hidden_activation, input_handler.output_activation, input_handler.apply_bias)
        '''
        print(f"Expected =\n{np.squeeze(test_Y)}\n")
        print(f"Output = {np.squeeze(O)}\n")
        print(f"Prediction =\n{np.squeeze(P)}\n")  # 0/1 Predictions
        print(f"Accuracy =  {np.mean((P == test_Y)) * 100}%\n")
        '''    
        acc.append(np.mean((P == test_Y)) * 100)
    accs.append(np.mean(acc))

    noise2 = 2 * np.random.rand(test_X.shape[0],test_X.shape[1]) - 1
    test_X_noise_2 = test_X + noise2

    acc = []
    for i in range(5):
        O, P = predict_multiclass(test_X_noise_2, parameters, input_handler.hidden_activation, input_handler.output_activation, input_handler.apply_bias)
        '''
        print(f"Expected =\n{np.squeeze(test_Y)}\n")
        print(f"Output = {np.squeeze(O)}\n")
        print(f"Prediction =\n{np.squeeze(P)}\n")  # 0/1 Predictions
        print(f"Accuracy =  {np.mean((P == test_Y)) * 100}%\n")
        '''    
        acc.append(np.mean((P == test_Y)) * 100)
    accs.append(np.mean(acc))

    # Graphics
    #fig = plt.figure(figsize=(10,5))
    bars = plt.bar(['training', 'noise (0,1)', 'noise (-1,1)'], accs, width=0.4)
    plt.title("Accuracy")
    plt.xlabel("Mode")
    plt.ylabel("Accuracy")
    plt.bar_label(bars)
    plt.show()

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
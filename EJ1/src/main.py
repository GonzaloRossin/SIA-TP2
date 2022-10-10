from cmath import exp
from perceptron import *
from data import *
from analyse_results import *
from plot import *
from input_handler import *
import json
import matplotlib as plt
import numpy as np


def main():
    
    with open('config.json', 'r') as f:
        json_file = json.load(f)
        input_handler = InputHandler(json_file)

    input, expected = data(input_handler.operation)
    #input, expected = data("xor")
    #input, expected = data(sys.argv[0])

    w, miss_l = perceptron(input, expected, input_handler.learning_rate, input_handler.num_epochs, input_handler.operation)
    plot(input, expected, w, input_handler.operation)

    print("weights: ", w)
    #print("Accuracy perceptron:", accuracy(expected, w))


    #keep weights, for graphic evolution of y=mx + b
    
if __name__ == "__main__":
    main()
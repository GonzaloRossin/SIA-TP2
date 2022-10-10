from cmath import exp
from perceptron import *
from data import *
from analyse_results import *
from plot import *
import matplotlib as plt
import numpy as np


input, expected = data("and")
#input, expected = data("xor")
#input, expected = data(sys.argv[0])

w, miss_l = perceptron(input, expected, 0.0001, 100)
plot(input, expected, w, "and")

print("weights: ", w)
#print("Accuracy perceptron:", accuracy(expected, w))


#keep weights, for graphic evolution of y=mx + b
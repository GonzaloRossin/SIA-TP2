from cmath import exp
from perceptron import *
from data import *
from analyse_results import *
import matplotlib as plt
import numpy as np


input, expected = data("and")
#input, expected = data("xor")
#input, expected = data(sys.argv[0])

theta, miss_l = perceptron(input, expected, 0.0001, 1000)

print("Accuracy perceptron:", accuracy(expected, theta))

from tkinter import W
import matplotlib.pyplot as plt
import numpy as np


def plot(input, expected, weights, title):
    plt.rcParams['figure.figsize'] = (6.0, 5.0)
    plt.title(title)
    plt.xlim([-2,2])
    plt.ylim([-1.5,1.5])
    #weights[0] += 0.00001
    #weights[1] += 0.00001
    #weights[2] += 0.00001
    x = np.arange(-2,3,0.1)
    y = ((-1 * weights[2] / weights[1]) * x) + (-1 * weights[0] / weights[1])
    for i in range(0, len(input)):
        if expected[i] == 1:
            plt.scatter(input[i][1], input[i][2], c="blue")
        else:
            plt.scatter(input[i][1], input[i][2], c="red")
    plt.plot(x, y)
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def readCSV(csv_filepath):
    df = pd.read_csv(csv_filepath)
    return df


def plotw(wOverT):
    i = 0
    iterations = []
    w1 = []
    w2 = []
    w3 = []
    for weight in wOverT:
        w1.append(weight[0])
        w2.append(weight[1])
        w3.append(weight[2])
        iterations.append(i)
        i += 1
    plt.plot(iterations, w1, label="w1")
    plt.plot(iterations, w2, label="w2")
    plt.plot(iterations, w3, label="w3")
    plt.legend()
    plt.show()


def plotError(errorVsT, iterations):
    i = 0
    iterationsList = []

    while i < iterations:
        iterationsList.append(i)
        i += 1

    plt.plot(iterationsList, errorVsT, label="error")
    plt.legend()
    plt.show()


class InputUtil:
    def __init__(self, csv_path):
        df = readCSV(csv_path)
        self.inputMatrix = np.zeros((len(df.x1), 5))
        self.weightMatrix = np.zeros((1, 4))
        for i in range(len(df.x1)):
            for j in range(5):
                if j == 0:
                    self.inputMatrix[i][j] = 1
                if j == 1:
                    self.inputMatrix[i][j] = df.x1[i]
                elif j == 2:
                    self.inputMatrix[i][j] = df.x2[i]
                elif j == 3:
                    self.inputMatrix[i][j] = df.x3[i]
                elif j == 4:
                    self.inputMatrix[i][j] = df.y[i]


    def getInputMatrix(self):
        return self.inputMatrix

    def getWeightMatrix(self):
        return self.weightMatrix

    def getTrainingSetByPercentage(self, percentage):

        totalRows = self.inputMatrix.shape[0]
        trainingSet = np.array([self.inputMatrix[0]])
        currentRows = 1

        while (currentRows / totalRows)*100 < percentage:
            row = np.array(self.inputMatrix[currentRows])
            trainingSet = np.append(trainingSet, [row], axis=0)
            currentRows += 1

        testSet = np.array([self.inputMatrix[currentRows]])
        currentRows += 1
        while currentRows < totalRows:
            row = np.array(self.inputMatrix[currentRows])
            testSet = np.append(trainingSet, [row], axis=0)
            currentRows += 1

        return trainingSet, testSet


import numpy
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


def calculateError(errorMatrix):
    i = 0
    x = []
    average = []
    while i < len(errorMatrix[0]):
        x.append(i)
        errorValues = []
        for errorList in errorMatrix:
            errorValues.append(errorList[i])
        arr = np.asarray(errorValues)
        average.append(sum(arr) / len(errorValues))
        i += 1

    aux = np.array(errorMatrix)
    standardDeviation= np.std(aux, 0)
    return x, average, standardDeviation


def plotError(errorVsT):
    x, average, standardDvt = calculateError(errorVsT)

    plt.plot(x, average, label="error")
    plt.fill_between(x, average -standardDvt, average + standardDvt, color="lightblue", label="error")
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

        while (currentRows / totalRows) * 100 < percentage:
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

    def splitInputFromResult(self, inputSet):
        trainingSet = np.copy(inputSet)
        resultVector = np.copy(inputSet)
        if inputSet.ndim > 1:
            trainingSet = np.delete(trainingSet, len(trainingSet[0]) - 1, 1)
        else:
            trainingSet = np.delete(trainingSet, len(trainingSet) - 1)

        for i in range(4):
            if inputSet.ndim > 1:
                resultVector = np.delete(resultVector, 0, 1)
            else:
                resultVector = np.delete(resultVector, 0)

        if trainingSet.ndim == 1:
            trainingSet = trainingSet[np.newaxis, :]
            resultVector = resultVector[np.newaxis, :]

        return trainingSet, resultVector

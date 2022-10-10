from unittest import result
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from ActivationType import ActivationType


def readCSV(csv_filepath):
    df = pd.read_csv(csv_filepath)
    return df

def getEtha():
    df = pd.read_csv("parameters.csv")
    return df['etha']

def getBeta():
    df = pd.read_csv("parameters.csv")
    return df['beta']

def getIterations():
    df = pd.read_csv("parameters.csv")
    return df['iterations']

def getIterations():
    df = pd.read_csv("parameters.csv")
    return df['iterations']


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
        
        np.random.shuffle(self.inputMatrix)
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

    def exportXYZModel(self, wvsT, perceptron):
        if os.path.exists("app.cpp"):
            os.remove("app.cpp")
        f = open('model.xyz', 'w')
        inputMatrix, resultVector = self.splitInputFromResult(self.inputMatrix)
        particleCount = inputMatrix.shape[0]
        for w in wvsT:
            f.write(str(particleCount) + '\n' + '\n')
            for inputVector in inputMatrix:
                h = np.dot(inputVector, w)
                result = perceptron.calculateO(h, perceptron.activationType)
                if perceptron.activationType != ActivationType.LINEAR:
                    result = perceptron.deNormalize(result)
                f.write(str(inputVector[1]) + ' ' + str(inputVector[2]) + ' ' + str(inputVector[3]) + ' ' + str(
                    result) + '\n')

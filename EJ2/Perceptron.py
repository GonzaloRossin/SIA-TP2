from locale import normalize
import math
import random
import numpy as np

import ActivationType
from Utils import readCSV
import SelectionType


class Perceptron:
    def __init__(self, inputMatrix, utils, weightVector, activationType):
        self.inputMatrix = inputMatrix
        self.weightVector = weightVector
        self.trainingInput = None
        self.resultVector = None
        self.utils = utils
        self.activationType = activationType

    def calculateO(self, inputSum, activationType, beta):
        if activationType == ActivationType.ActivationType.LINEAR:
            return inputSum
        elif activationType == ActivationType.ActivationType.SIGMOID:
            return math.tanh(beta * inputSum)

    def calculateError(self, inputMatrix, weightVector, activationType, beta):
        error = 0
        for i in range(inputMatrix.shape[0]):
            result = self.resultVector[i]
            if activationType == ActivationType.ActivationType.SIGMOID:
                result = self.normalize(result)
            h = np.dot(inputMatrix[i], weightVector)
            error += (result - self.calculateO(h, activationType, beta)) ** 2
        return 0.5 * error[0]

    def normalize(self, result):
        return (2 * (result - np.min(self.resultVector)) / (np.max(self.resultVector) - np.min(self.resultVector))) - 1

    def evaluateGdiff(self, O, beta):
        if ActivationType.ActivationType.LINEAR == self.activationType:
            return 1
        elif ActivationType.ActivationType.SIGMOID == self.activationType:
            return beta * (1 - O)

    def calculateWeights(self, activationType, weight_vector, beta, N):
        sumResultVector = []
        for j in range(len(self.trainingInput[0])):
            sumResult = 0
            for i in range(self.trainingInput.shape[0]):
                result = self.resultVector[i]
                h = np.dot(self.trainingInput[i], weight_vector)
                gDiff = self.evaluateGdiff(self.calculateO(h, self.activationType, beta), beta)
                Ei = self.trainingInput[i][j]
                if activationType == ActivationType.ActivationType.SIGMOID:
                    sumResult += (self.normalize(result) - self.calculateO(h, self.activationType, beta) * gDiff * Ei)
                else:
                    sumResult += (self.normalize(result) - self.calculateO(h, self.activationType, beta) * gDiff * Ei)
            sumResultVector.append(sumResult[0])

        for i in range(len(weight_vector)):
            weight_vector[i] = weight_vector[i] + N * sumResultVector[i]

        return weight_vector

    def readBetaparameter(self):
        df = readCSV('parameters.csv')
        return float(df.beta[0])

    def readWeirdNparameter(self):
        df = readCSV('parameters.csv')
        return float(df.weird_N[0])

    def trainPerceptron(self, weight_vector, upper_limit, selectionType):
        beta = self.readBetaparameter()
        N = self.readWeirdNparameter()
        wOverTime = []
        errorVsT = []
        i = 0
        w = weight_vector
        error_min = 1000000
        w_min = None
        # self.getMinMaxValues(self.inputMatrix)
        if selectionType == SelectionType.SelectionType.RANDOM:
            while error_min > 0 and i < upper_limit:
                pickInput = random.choice(self.inputMatrix)
                w = self.calculateWeights(self.activationType, weight_vector, beta, N)
                error = self.calculateError(self.inputMatrix, w, self.activationType, beta)
                if error < error_min:
                    error_min = error
                    w_min = w
                wOverTime.append(w)
                errorVsT.append(error)
                i += 1
        else:
            while error_min > 0 and i < upper_limit:

                np.random.shuffle(self.inputMatrix)
                self.trainingInput, self.resultVector = self.utils.splitInputFromResult(self.inputMatrix)
                self.calculateWeights(self.activationType, w, beta, N)
                error = self.calculateError(self.trainingInput, w, self.activationType, beta)
                wOverTime.append(w)
                errorVsT.append(error)
                if error < error_min:
                    error_min = error
                    w_min = w
                i += 1

        return w_min, wOverTime, errorVsT, error_min

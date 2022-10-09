from locale import normalize
import math
import random
import numpy as np

import ActivationType
from Utils import readCSV
import SelectionType


class Perceptron:
    def __init__(self, inputMatrix, utils, weightVector, activationType, etha, beta):
        self.inputMatrix = inputMatrix
        self.weightVector = weightVector
        self.trainingInput = None
        self.resultVector = None
        self.etha = etha
        self.beta = beta
        self.utils = utils
        self.activationType = activationType
        self.min, self.max = self.calculateMinMax()

    def calculateO(self, h, activationType):
        if activationType == ActivationType.ActivationType.LINEAR:
            return h
        elif activationType == ActivationType.ActivationType.SIGMOID_TANH:
            return math.tanh(self.beta * h)
        elif activationType == ActivationType.ActivationType.SIGMOID_LOGISTIC:
            return 1/(1 + math.pow(math.e, -2*self.beta*h))

    def calculateError(self, inputMatrix, resultVector, weightVector, activationType):
        error = 0
        for i in range(inputMatrix.shape[0]):
            result = resultVector[i][0]
            h = np.dot(inputMatrix[i], weightVector)
            if activationType != ActivationType.ActivationType.LINEAR:
                error += (result - self.deNormalize(self.calculateO(h, activationType))) ** 2
            else:
                error += (result - self.calculateO(h, activationType)) ** 2
        return 0.5 * error

    def calculateMinMax(self):
        inputMatrix, resultVector = self.utils.splitInputFromResult(self.utils.inputMatrix)
        return min(resultVector)[0], max(resultVector)[0]

    def normalize(self, result):
        if self.activationType == ActivationType.ActivationType.SIGMOID_TANH:
            return (2 * (result - self.min) / (self.max - self.min)) - 1
        else:
            return (result - self.min)/(self.max-self.min)
    
    def deNormalize(self, result):
        if self.activationType == ActivationType.ActivationType.SIGMOID_TANH:
            return ((result+1) * (self.max - self.min) * 0.5) + self.min
        else:
            return result * (self.max-self.min) + self.min

    def evaluateGdiff(self, O):
        if ActivationType.ActivationType.LINEAR == self.activationType:
            return 1
        elif ActivationType.ActivationType.SIGMOID_TANH == self.activationType:
            return self.beta * (1 - O**2)
        elif ActivationType.ActivationType.SIGMOID_LOGISTIC == self.activationType:
            return 2*self.beta*O*(1-O)

    def calculateWeights(self, activationType, weight_vector):
        sumResultVector = []
        for j in range(len(self.trainingInput[0])):
            sumResult = 0
            for i in range(self.trainingInput.shape[0]):
                result = self.resultVector[i]
                h = np.dot(self.trainingInput[i], weight_vector)
                O = self.calculateO(h, self.activationType)
                gDiff = self.evaluateGdiff(O)
                Ei = self.trainingInput[i][j]
                if activationType != ActivationType.ActivationType.LINEAR:
                    sumResult += ((self.normalize(result) - O) * gDiff * Ei)
                else:
                    sumResult += ((result - O) * gDiff * Ei)

            sumResultVector.append(sumResult[0])

        for i in range(len(weight_vector)):
            weight_vector[i] = weight_vector[i] + self.etha * sumResultVector[i]

        return weight_vector

    def trainPerceptron(self, weight_vector, upper_limit, selectionType):
        errorVsT = []
        i = 0
        w = weight_vector
        error_min = 1000000
        w_min = None
    
        while error_min > 0 and i < upper_limit:
            if selectionType == SelectionType.SelectionType.RANDOM:
                pickInput = random.choice(self.inputMatrix)
                self.trainingInput, self.resultVector = self.utils.splitInputFromResult(pickInput)
            else:
                np.random.shuffle(self.inputMatrix)
                self.trainingInput, self.resultVector = self.utils.splitInputFromResult(self.inputMatrix)
            w = self.calculateWeights(self.activationType, w)
            error = self.calculateError(self.trainingInput, self.resultVector,  w, self.activationType)
            if error < error_min:
                error_min = error
                w_min = w
            errorVsT.append(error)
            i += 1

        return w_min, errorVsT, error_min

from locale import normalize
import math
import random
import numpy as np

import ActivationType
from Utils import readCSV
import SelectionType


class Perceptron:
    def __init__(self, inputMatrix, weightVector, perceptronType):
        self.inputMatrix = inputMatrix
        self.weightVector = weightVector
        self.perceptronType = perceptronType
        self.maxResult = 0
        self.minResult = 0

    def calculateO(self, inputSum, perceptronType, beta):
        if perceptronType == ActivationType.ActivationType.LINEAR:
            return inputSum
        elif perceptronType == ActivationType.ActivationType.SIGMOID:
            return math.tanh(beta * inputSum)

    def calculateError(self, inputMatrix, weightVector, perceptronType, beta):
        toRet = 0
        for i in range(inputMatrix.shape[0]):
            result = inputMatrix[i][len(inputMatrix[i]) - 1]
            if perceptronType == ActivationType.ActivationType.SIGMOID:
                result = self.normalize(result)
            inputSum = self.getInputSum(inputMatrix[i], weightVector)
            toRet += (result - self.calculateO(inputSum, perceptronType, beta)) ** 2
        return 0.5 * toRet

    def getMinMaxValues(self):
        self.getMaxValue()
        self.getMinValue()

    def getMaxValue(self):
        maxValue = 0
        for input_vector in self.inputMatrix:
            if input_vector[4] > maxValue:
                maxValue = input_vector[4]

        self.maxResult = maxValue

    def getMinValue(self):
        minValue = self.inputMatrix[0][4]
        for i in range(self.inputMatrix.shape[0] - 1):
            if self.inputMatrix[i + 1][4] < minValue:
                minValue = self.inputMatrix[i + 1][4]

        self.minResult = minValue

    def normalize(self, result):
        return (2 * (result - self.minResult) / (self.maxResult - self.minResult)) - 1

    def getInputSum(self, input_vector, weight_vector):
        Sum = 0
        for i in range(len(input_vector) - 1):
            weight = weight_vector[i]
            inputValue = input_vector[i]
            Sum += weight * inputValue
        return Sum

    def calculateWeights(self, input_vectorList, input_vector, weight_vector, beta, N):
        ExpectedVsResultSum = 0
        newWeights = []
        sumatoryVector = []
        for i in range(len(input_vector) - 1):
            inputValue = input_vector[i]
            for j in range(len(input_vectorList)):
                result = input_vectorList[j][len(input_vector) - 1]
                if self.perceptronType == ActivationType.ActivationType.SIGMOID:
                    result = self.normalize(result)
                inputSum = self.getInputSum(input_vectorList[j], weight_vector)
                ExpectedVsResult = (result - self.calculateO(inputSum, self.perceptronType, beta)) * inputValue
                if self.perceptronType == ActivationType.ActivationType.SIGMOID:
                    ExpectedVsResult = ExpectedVsResult * beta * (1 - self.calculateO(
                        self.getInputSum(input_vector, weight_vector),
                        self.perceptronType, beta))
                ExpectedVsResultSum += ExpectedVsResult

            sumatoryVector.append(ExpectedVsResultSum)

        for i in range(len(input_vector) - 1):
            weightDelta = N * sumatoryVector[i]

            newWeights.append(weight_vector[i] + weightDelta)

        return newWeights

    def readBetaparameter(self):
        df = readCSV('parameters.csv')
        return float(df.beta[0])

    def readWeirdNparameter(self):
        df = readCSV('parameters.csv')
        return float(df.weird_N[0])

    def trainPerceptron(self, input_vectorList, weight_vector, upper_limit, selectionType):
        beta = self.readBetaparameter()
        N = self.readWeirdNparameter()
        wOverTime = []
        errorVsT = []
        i = 0
        j = 0
        w = weight_vector
        error_min = 1000000
        w_min = None
        self.getMinMaxValues()
        inputRows = input_vectorList.shape[0]
        if selectionType ==  SelectionType.SelectionType.RANDOM:
            while error_min > 0 and i < upper_limit:
                pickInput = random.choice(input_vectorList)
                h = self.getInputSum(pickInput, w)
                O = self.calculateO(h, self.perceptronType, beta)
                w = self.calculateWeights(input_vectorList, pickInput, w, beta, N)
                error = self.calculateError(input_vectorList, w, self.perceptronType, beta)
                if error < error_min:
                    error_min = error
                    w_min = w
                wOverTime.append(w)
                errorVsT.append(error)
                i += 1
        else:
            shuffledTrainingSet = input_vectorList
            error = None
            while error_min > 0 and i < upper_limit:

                np.random.shuffle(shuffledTrainingSet)
                while error_min > 0 and j < inputRows:
                    pickInput = shuffledTrainingSet[i]
                    h = self.getInputSum(pickInput, w)
                    O = self.calculateO(h, self.perceptronType, beta)
                    w = self.calculateWeights(input_vectorList, pickInput, w, beta, N)
                    error = self.calculateError(input_vectorList, w, self.perceptronType, beta)
                    j += 1
                
                if error < error_min:
                    error_min = error
                    w_min = w        
                wOverTime.append(w)
                errorVsT.append(error)
                i += 1

        return w_min, wOverTime, errorVsT, error_min, i

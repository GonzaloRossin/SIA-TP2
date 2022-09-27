import math
import random

import ActivationType
from Utils import readCSV


def calculateO(inputSum, perceptronType, beta):
    if perceptronType == ActivationType.ActivationType.LINEAR:
        return inputSum
    elif perceptronType == ActivationType.ActivationType.SIGMOID:
        return math.tanh(beta * inputSum)


def calculateError(inputMatrix, Oresult):
    toRet = 0
    for i in range(inputMatrix.shape[0]):
        result = inputMatrix[i][len(inputMatrix[i]) - 1]
        toRet += (result - Oresult) ** 2
    return 0.5 * toRet


def getInputSum(input_vector, weight_vector):
    Sum = 0
    for i in range(len(input_vector) - 1):
        weight = weight_vector[i]
        inputValue = input_vector[i]
        Sum += weight * inputValue
    return Sum


def calculateWeights(input_vectorList, input_vector, weight_vector, perceptronType, beta, N):
    ExpectedVsResultSum = 0
    newWeights = []
    sumatoryVector = []
    for i in range(len(input_vector) - 1):
        inputValue = input_vector[i]
        for j in range(len(input_vectorList)):
            result = input_vectorList[j][len(input_vector) - 1]
            inputSum = getInputSum(input_vectorList[j], weight_vector)
            ExpectedVsResult = (result - calculateO(inputSum, perceptronType, beta)) * inputValue
            if perceptronType == ActivationType.ActivationType.SIGMOID:
                ExpectedVsResult = ExpectedVsResult * beta * (1 - calculateO(getInputSum(input_vector, weight_vector),
                                                                             perceptronType, beta))
            ExpectedVsResultSum += ExpectedVsResult

        sumatoryVector.append(ExpectedVsResultSum)

    for i in range(len(input_vector) - 1):
        weightDelta = N * sumatoryVector[i]

        newWeights.append(weight_vector[i] + weightDelta)

    return newWeights


def readBetaparameter():
    df = readCSV('parameters.csv')
    return float(df.beta[0])


def readWeirdNparameter():
    df = readCSV('parameters.csv')
    return float(df.weird_N[0])


def trainPerceptron(input_vectorList, weight_vector, upper_limit, perceptron_type):
    beta = readBetaparameter()
    N = readWeirdNparameter()
    wOverTime = []
    errorVsT = []
    i = 0
    w = weight_vector
    error_min = 1000000
    while error_min > 0 and i < upper_limit:
        pickInput = random.choice(input_vectorList)
        h = getInputSum(pickInput, w)
        O = calculateO(h, perceptron_type, beta)
        w = calculateWeights(input_vectorList, pickInput, w, perceptron_type, beta, N)
        wOverTime.append(w)
        error = calculateError(input_vectorList, O)
        errorVsT.append(error)
        if error < error_min:
            error_min = error
            w_min = w
        i += 1

    return w, wOverTime, errorVsT

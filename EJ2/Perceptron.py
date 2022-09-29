import math
import random

import ActivationType
from Utils import readCSV
import SelectionType


def calculateO(inputSum, perceptronType, beta):
    if perceptronType == ActivationType.ActivationType.LINEAR:
        return inputSum
    elif perceptronType == ActivationType.ActivationType.SIGMOID:
        return math.tanh(beta * inputSum)


def calculateError(inputMatrix, weightVector, perceptronType, beta):
    toRet = 0
    for i in range(inputMatrix.shape[0]):
        result = inputMatrix[i][len(inputMatrix[i]) - 1]
        inputSum = getInputSum(inputMatrix[i], weightVector)
        toRet += (result - calculateO(inputSum, perceptronType, beta)) ** 2
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


def trainPerceptron(input_vectorList, weight_vector, upper_limit, perceptron_type, selectionType):
    beta = readBetaparameter()
    N = readWeirdNparameter()
    wOverTime = []
    errorVsT = []
    i = 0
    w = weight_vector
    error_min = 1000000
    w_min = None
    inputRows = input_vectorList.shape[0]
    if SelectionType.SelectionType.RANDOM:
        while error_min > 0 and i < upper_limit:
            pickInput = random.choice(input_vectorList)
            h = getInputSum(pickInput, w)
            O = calculateO(h, perceptron_type, beta)
            w = calculateWeights(input_vectorList, pickInput, w, perceptron_type, beta, N)
            error = calculateError(input_vectorList, w, perceptron_type, beta)
            if error < error_min:
                error_min = error
                w_min = w
            wOverTime.append(w)
            errorVsT.append(error)
            i += 1
    else:
     while error_min > 0 and i < inputRows:
        pickInput = input_vectorList[i]
        h = getInputSum(pickInput, w)
        O = calculateO(h, perceptron_type, beta)
        w = calculateWeights(input_vectorList, pickInput, w, perceptron_type, beta, N)
        error = calculateError(input_vectorList, w, perceptron_type, beta)
        if error < error_min:
            error_min = error
            w_min = w
        wOverTime.append(w)
        errorVsT.append(error)
        i += 1

    return w, wOverTime, errorVsT, error_min

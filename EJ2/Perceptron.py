import math

import ActivationType
from Utils import readCSV


def calculateO(inputSum, perceptronType, beta):
    if perceptronType == ActivationType.ActivationType.LINEAR:
        return inputSum
    elif perceptronType == ActivationType.ActivationType.SIGMOID:
        return math.tanh(beta * inputSum)


def calculateError(inputMatrix, result_vector, weightVector, perceptronType, beta):
    toRet = 0
    for i in range(inputMatrix.shape[0]):
        result = result_vector[i]
        inputSum = getInputSum(inputMatrix[i], weightVector[i])
        toRet += (result - calculateO(inputSum, perceptronType, beta)) ** 2
    return 0.5 * toRet


def calculateWeightDelta(input_vectorList, input_vectorIndex, result_vector, weight_vector, perceptronType, beta, N):
    ExpectedVsResultSum = 0
    input_vector = input_vectorList[input_vectorIndex]
    weightDeltaVector = []
    for i in range(len(input_vectorList)):
        result = result_vector[i]
        inputSum = getInputSum(input_vectorList[i], weight_vector)
        ExpectedVsResultSum += result - calculateO(inputSum, perceptronType, beta)

    for inputValue in input_vector:

        weightDelta = N * ExpectedVsResultSum * inputValue

        if perceptronType == ActivationType.ActivationType.SIGMOID:
            weightDelta *= beta * (1 - calculateO(getInputSum(input_vector, weight_vector), perceptronType, beta))

        weightDeltaVector.append(weightDelta)


def getInputSum(input_vector, weight_vector):
    Sum = 0
    for i in range(len(input_vector)):
        Sum += weight_vector[i] * input_vector[i]
    return Sum


def readBetaparameter():
    df = readCSV('parameters.csv')
    return df.beta


def readWeirdNparameter():
    df = readCSV('parameters.csv')
    return df.weird_N

import math

import ActivationType.ActivationType
from Utils import readCSV


def calculateError(input_vector, result_vector, weight_vector, perceptronType):
        toRet = 0
        for i in range(len(input_vector)):
            result = result_vector[i]
            toRet += (result - calculateO(input_vector[i], weight_vector[i], perceptronType)) ** 2
        return 0.5 * toRet


def calculateDeltaW(inputVectorList, weight, perceptronType, result_vector):
    weird_N = readWeirdNparameter()
    Sum = 0
    delta_weight = []
    if perceptronType == ActivationType.ActivationType.LINEAR:
        for i in range(len(inputVectorList)):
            Sum += (result_vector[i] - calculateO(inputVectorList[i], weight, perceptronType))
        for j in range(len(inputVectorList[0])):
            delta_weight.append(weird_N * Sum * inputVectorList[j])
        return delta_weight
    elif perceptronType == ActivationType.ActivationType.SIGMOID:
        return 1


def calculateO(input_vector, weight_vector, perceptronType):
    returnSum = 0
    for i in range(len(input_vector)):
        returnSum += input_vector[i] * weight_vector[i]
    if perceptronType == ActivationType.ActivationType.LINEAR:
        return returnSum
    elif perceptronType == ActivationType.ActivationType.SIGMOID:
        beta = readBetaparameter()

        return 1/(math.e**(2*beta*returnSum))

def readBetaparameter():
    df = readCSV('parameters.csv')
    return df.beta
def readWeirdNparameter():
    df = readCSV('parameters.csv')
    return df.weird_N

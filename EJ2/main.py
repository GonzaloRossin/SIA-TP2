from pyexpat import model
from unittest import TestResult, result
from SelectionType import SelectionType
from Utils import *
from Perceptron import trainPerceptron
import ActivationType

inputUtil = InputUtil('TP2-ej2-conjunto.csv')

inputMatrix = inputUtil.getInputMatrix()
weightMatrix = inputUtil.getWeightMatrix()
trainingSet, testSet = inputUtil.getTrainingSetByPercentage(60)
iterations = 100000

weights, wVsIteration, errorVsIteration, error_min = trainPerceptron(trainingSet, weightMatrix[0], iterations,
                                                          ActivationType.ActivationType.LINEAR, SelectionType.EPOCA)

print('E1   |  E2    | E3    ||| RESULT                |   EXPECTED')
for test_vector in testSet:
    modelResult = weights[1] * test_vector[1] + weights[2] * test_vector[2] + weights[3] * test_vector[3]
    print(test_vector[1], ' | ', test_vector[2], ' | ', test_vector[3], ' ||| ', modelResult, ' | ', test_vector[4])
print('error_min= ',error_min)

plotw(wVsIteration)

plotError(errorVsIteration, iterations)

from SelectionType import SelectionType
from Utils import *
from Perceptron import Perceptron
import ActivationType

inputUtil = InputUtil('TP2-ej2-conjunto.csv')

inputMatrix = inputUtil.getInputMatrix()
weightMatrix = inputUtil.getWeightMatrix()
trainingSet, testSet = inputUtil.getTrainingSetByPercentage(60)
iterations = 9000
perceptron = Perceptron(inputMatrix, weightMatrix[0], ActivationType.ActivationType.LINEAR)

weights, wVsIteration, errorVsIteration, error_min = perceptron.trainPerceptron(trainingSet, weightMatrix[0]
                                                                                , iterations
                                                                                , SelectionType.EPOCA)

print(weights)
print('E1   |  E2    | E3    ||| RESULT                |   EXPECTED')
for test_vector in testSet:
    modelResult = weights[0] + weights[1] * test_vector[1] + weights[2] * test_vector[2] + weights[3] * test_vector[3]
    if perceptron.perceptronType == ActivationType.ActivationType.SIGMOID:
        print(test_vector[1], ' | ', test_vector[2], ' | ', test_vector[3], ' ||| ', modelResult, ' | ', test_vector[4])
    else:
        print(test_vector[1], ' | ', test_vector[2], ' | ', test_vector[3], ' ||| ', modelResult, ' | ', test_vector[4])
print('error_min= ', error_min)

plotw(wVsIteration)

plotError(errorVsIteration)

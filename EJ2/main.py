from SelectionType import SelectionType
from Utils import *
from Perceptron import Perceptron
import ActivationType

inputUtil = InputUtil('TP2-ej2-conjunto.csv')

trainingSet, testSet = inputUtil.getTrainingSetByPercentage(60)
testSet, testResults = inputUtil.splitInputFromResult(testSet)
iterations = 2000
perceptron = Perceptron(trainingSet, inputUtil, inputUtil.getWeightMatrix(), ActivationType.ActivationType.SIGMOID_LOGISTIC)

weights, wVsIteration, errorVsIteration, error_min = perceptron.trainPerceptron(inputUtil.getWeightMatrix()[0]
                                                                                , iterations
                                                                                , SelectionType.EPOCA)
print(weights)
print('E1   |  E2    | E3    ||| RESULT                |   EXPECTED')
for i in range(len(testSet)):
    modelResult = np.dot(weights, testSet[i])
    if perceptron.activationType != ActivationType.ActivationType.LINEAR:
        modelResult = perceptron.deNormalize(modelResult)
    print(testSet[i][1], ' | ', testSet[i][2], ' | ', testSet[i][3], ' ||| ', modelResult, ' | ', testResults[i][0])

plotError(errorVsIteration)

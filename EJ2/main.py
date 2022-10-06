import copy

from SelectionType import SelectionType
from Utils import *
from Perceptron import Perceptron
import ActivationType

inputUtil = InputUtil('TP2-ej2-conjunto.csv')

trainingSet, testSet = inputUtil.getTrainingSetByPercentage(60)
testSet, testResults = inputUtil.splitInputFromResult(testSet)
errorListAcumulator = []

etha = 0.01
iterations = 50000
for i in range(6):
    inputCopy = copy.deepcopy(inputUtil)
    perceptron = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                            , ActivationType.ActivationType.LINEAR, etha, 1)
    weights, errorList, error_min = perceptron.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                               , iterations
                                                               , SelectionType.RANDOM)
    errorListAcumulator.append(errorList)

plotError(errorListAcumulator)
print('E1   |  E2    | E3    ||| RESULT                |   EXPECTED')
for i in range(len(testSet)):
    modelResult = np.dot(weights, testSet[i])
    if perceptron.activationType != ActivationType.ActivationType.LINEAR:
        modelResult = perceptron.deNormalize(modelResult)
    print(testSet[i][1], ' | ', testSet[i][2], ' | ', testSet[i][3], ' ||| ', modelResult, ' | ', testResults[i][0])

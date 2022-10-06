import copy

from SelectionType import SelectionType
from Utils import *
from Perceptron import Perceptron
import ActivationType

inputUtil = InputUtil('TP2-ej2-conjunto.csv')

trainingSet, testSet = inputUtil.getTrainingSetByPercentage(60)
testSet, testResults = inputUtil.splitInputFromResult(testSet)
errorListAcumulator = []

etha = 0.001
iterations = 5000
beta = 1
for i in range(1):
    inputCopy = copy.deepcopy(inputUtil)
    perceptron = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                            , ActivationType.ActivationType.SIGMOID_LOGISTIC, etha, beta)
    weights, errorList, error_min = perceptron.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                               , iterations
                                                               , SelectionType.EPOCA)
    errorListAcumulator.append(errorList)

print(weights)
plotError(errorListAcumulator)
print('E0  | E1   |  E2    | E3    ||| RESULT                |   EXPECTED')
for i in range(len(testSet)):
    modelResult = perceptron.calculateO( np.dot(weights, testSet[i]),perceptron.activationType)
    if perceptron.activationType != ActivationType.ActivationType.LINEAR:
        modelResult = perceptron.deNormalize(modelResult)
    print(testSet[i][0],'  |', testSet[i][1], '| ', testSet[i][2], ' | ', testSet[i][3], ' ||| ', modelResult, ' | ', testResults[i][0])

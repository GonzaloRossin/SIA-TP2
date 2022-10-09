import copy

from SelectionType import SelectionType
from Utils import *
from Perceptron import Perceptron
import ActivationType
import matplotlib.pyplot as plt


def plotBestProportion(inputUtil):
    trainingSetProportions = [10, 20, 30, 40, 50, 60, 70, 80]
    beta = 1
    etha = 0.001
    iterations = 7000
    
    resultsTanH = []
    resultsLogistic = []

    for trainingPercentage in trainingSetProportions:
        trainingSet, testSet = inputUtil.getTrainingSetByPercentage(trainingPercentage)
        testSet, testResults = inputUtil.splitInputFromResult(testSet)
        inputCopy = copy.deepcopy(inputUtil)
        perceptronTanh = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                            , ActivationType.ActivationType.SIGMOID_TANH, etha, beta)
        perceptronLogistic = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                            , ActivationType.ActivationType.SIGMOID_LOGISTIC, etha, beta)
        weightsTanh, errorList, error_min = perceptronTanh.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                               , iterations
                                                               , SelectionType.EPOCA)
        weightsLogistic, errorList, error_min = perceptronLogistic.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                               , iterations
                                                               , SelectionType.EPOCA)

        resultsTanH.append(perceptronTanh.calculateError(testSet, testResults, weightsTanh, perceptronTanh.activationType))
        resultsLogistic.append(perceptronLogistic.calculateError(testSet, testResults, weightsLogistic, perceptronLogistic.activationType))

    plt.bar(trainingSetProportions, resultsTanH, width=0.8)
    plt.show()
    plt.bar(trainingSetProportions, resultsLogistic, width=0.8)
    plt.show()

'''inputUtil = InputUtil('TP2-ej2-conjunto.csv')
trainingSet, testSet = inputUtil.getTrainingSetByPercentage(50)
testSet, testResults = inputUtil.splitInputFromResult(testSet)
errorListAcumulator = []

etha = 0.0001
iterations = 10000
beta = 1
for i in range(5):
    inputCopy = copy.deepcopy(inputUtil)
    perceptron = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                            , ActivationType.ActivationType.SIGMOID_TANH, etha, beta)
    weights, errorList, error_min = perceptron.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                               , iterations
                                                               , SelectionType.EPOCA)
    errorListAcumulator.append(errorList)


plotError(errorListAcumulator)'''
#print(perceptron.deNormalize(perceptron.calculateError(testSet, testResults, weights, perceptron.activationType)))
inputUtil = InputUtil('TP2-ej2-conjunto.csv')
plotBestProportion(inputUtil)

import copy
from statistics import mean

from matplotlib import pyplot as plt

from Perceptron import Perceptron
import ActivationType
from Utils import *


def plotBestProportion(inputUtil, etha, beta, iterations):
    trainingSetProportions = [10, 20, 30, 40, 50, 60, 70, 80]
    resultsMap = {}
    averageListTanh = []
    averageListLogistic = []

    for proportion in trainingSetProportions:
        resultsMap[proportion] = {ActivationType.SIGMOID_LOGISTIC: [],
                                  ActivationType.SIGMOID_TANH: []}

    for i in range(5):
        for trainingPercentage in trainingSetProportions:
            trainingSet, testSet = inputUtil.getTrainingSetByPercentage(trainingPercentage)
            testSet, testResults = inputUtil.splitInputFromResult(testSet)
            inputCopy = copy.deepcopy(inputUtil)
            perceptronTanh = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                                        , ActivationType.SIGMOID_TANH, etha, beta)
            perceptronLogistic = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                                            , ActivationType.SIGMOID_LOGISTIC, etha, beta)
            weightsTanh, _, _ = perceptronTanh.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                               , iterations
                                                               , SelectionType.EPOCA)
            weightsLogistic, _, _ = perceptronLogistic.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                                       , iterations
                                                                       , SelectionType.EPOCA)

            resultsMap[trainingPercentage][perceptronTanh.activationType].append(
                perceptronTanh.calculateError(testSet, testResults, weightsTanh, perceptronTanh.activationType))
            resultsMap[trainingPercentage][perceptronLogistic.activationType].append(
                perceptronLogistic.calculateError(testSet, testResults, weightsLogistic,
                                                  perceptronLogistic.activationType))

    for proportion in trainingSetProportions:
        averageListLogistic.append(mean(resultsMap[proportion][ActivationType.SIGMOID_LOGISTIC]))
        averageListTanh.append(mean(resultsMap[proportion][ActivationType.SIGMOID_TANH]))

    plt.bar(trainingSetProportions, averageListTanh, width=1)
    plt.show()
    plt.bar(trainingSetProportions, averageListLogistic, width=1)
    plt.show()


def calculateError(errorMatrix):
    i = 0
    x = []
    average = []
    maxValues = []
    minValues = []
    while i < len(errorMatrix[0]):
        x.append(i)
        errorValues = []
        for errorList in errorMatrix:
            errorValues.append(errorList[i])
        arr = np.asarray(errorValues)
        maxValues.append(max(arr))
        average.append(sum(arr) / len(errorValues))
        minValues.append(min(arr))
        i += 1

    return x, average, minValues, maxValues


def getErrorData(inputUtil, etha, beta, iterations, trainingPercentage, activationType, selectionType):
    errorListAcumulator = []

    for i in range(5):
        trainingSet, _ = inputUtil.getTrainingSetByPercentage(trainingPercentage)
        inputCopy = copy.deepcopy(inputUtil)
        perceptron = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                                , activationType, etha, beta)
        _, errorList, _ = perceptron.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                        , iterations
                                                        , selectionType)
        errorListAcumulator.append(errorList)

    return errorListAcumulator


def plotError(inputUtil, etha, beta, iterations, trainingPercentage, activationType, selectionType):
    x, average, minValues, maxValues = calculateError(getErrorData(inputUtil, etha, beta, iterations, trainingPercentage
                                                          , activationType, selectionType))

    plt.plot(x, average, label="error")
    plt.fill_between(x, minValues, maxValues, color="lightblue", label="error")
    plt.legend()
    plt.show()

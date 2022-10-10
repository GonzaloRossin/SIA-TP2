import copy
from statistics import mean
from Perceptron import Perceptron
from SelectionType import SelectionType
import ActivationType
from Utils import *


def plotBestProportion(inputUtil):
    trainingSetProportions = [10, 20, 30, 40, 50, 60, 70, 80]
    beta = 1
    etha = 0.0001
    iterations = 8000
    resultsMap = {}
    averageListTanh = []
    averageListLogistic = []

    for proportion in trainingSetProportions:
        resultsMap[proportion] = {ActivationType.ActivationType.SIGMOID_LOGISTIC: [],
                                  ActivationType.ActivationType.SIGMOID_TANH: []}

    for i in range(5):
        for trainingPercentage in trainingSetProportions:
            trainingSet, testSet = inputUtil.getTrainingSetByPercentage(trainingPercentage)
            testSet, testResults = inputUtil.splitInputFromResult(testSet)
            inputCopy = copy.deepcopy(inputUtil)
            perceptronTanh = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                                        , ActivationType.ActivationType.SIGMOID_TANH, etha, beta)
            perceptronLogistic = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                                            , ActivationType.ActivationType.SIGMOID_LOGISTIC, etha, beta)
            weightsTanh, _, _, _ = perceptronTanh.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                                  , iterations
                                                                  , SelectionType.EPOCA)
            weightsLogistic, _, _, _ = perceptronLogistic.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                                          , iterations
                                                                          , SelectionType.EPOCA)

            resultsMap[trainingPercentage][perceptronTanh.activationType].append(
                perceptronTanh.calculateError(testSet, testResults, weightsTanh, perceptronTanh.activationType))
            resultsMap[trainingPercentage][perceptronLogistic.activationType].append(
                perceptronLogistic.calculateError(testSet, testResults, weightsLogistic,
                                                  perceptronLogistic.activationType))

    for proportion in trainingSetProportions:
        averageListLogistic.append(mean(resultsMap[proportion][ActivationType.ActivationType.SIGMOID_LOGISTIC]))
        averageListTanh.append(mean(resultsMap[proportion][ActivationType.ActivationType.SIGMOID_TANH]))

    plt.bar(trainingSetProportions, averageListTanh, width=1)
    plt.show()
    plt.bar(trainingSetProportions, averageListLogistic, width=1)
    plt.show()


def calculateError(errorMatrix):
    i = 0
    x = []
    average = []
    while i < len(errorMatrix[0]):
        x.append(i)
        errorValues = []
        for errorList in errorMatrix:
            errorValues.append(errorList[i])
        arr = np.asarray(errorValues)
        average.append(sum(arr) / len(errorValues))
        i += 1

    standardDeviation= np.std(np.array(errorMatrix))
    return x, average, standardDeviation

def getErrorData(inputUtil, etha, beta, iterations):
    trainingSet, _ = inputUtil.getTrainingSetByPercentage(30)
    errorListAcumulator = []

    for i in range(5):
        inputCopy = copy.deepcopy(inputUtil)
        perceptron = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                            , ActivationType.ActivationType.SIGMOID_TANH, etha, beta)
        _, errorList,_, _ = perceptron.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                               , iterations
                                                               , SelectionType.EPOCA)
        errorListAcumulator.append(errorList)
    
    return errorListAcumulator

def plotError(inputUtil):
    x, average, standardDvt = calculateError(getErrorData(inputUtil))

    plt.plot(x, average, label="error")
    plt.fill_between(x, average - standardDvt, average + standardDvt, color="lightblue", label="error")
    plt.legend()
    plt.show()

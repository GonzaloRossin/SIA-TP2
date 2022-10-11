from cProfile import label
import copy
from statistics import mean

from matplotlib import pyplot as plt

from Perceptron import Perceptron
import ActivationType
from Utils import *


def plotBestProportion(inputUtil, etha, beta, iterations):
    trainingSetProportions = [10, 20, 30, 40, 50, 60, 70, 80]
    resultsMap = {}
    averageListTanhTest = []
    averageListTanhTraining = []
    averageListLogisticTest = []
    averageListLogisticTraining = []

    for proportion in trainingSetProportions:
        resultsMap[proportion] = {ActivationType.SIGMOID_LOGISTIC: {'training': [], 'test': []},
                                  ActivationType.SIGMOID_TANH: {'training': [], 'test':[]}}

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

            resultsMap[trainingPercentage][perceptronTanh.activationType]['test'].append(
                perceptronTanh.calculateError(testSet, testResults, weightsTanh, perceptronTanh.activationType))
            resultsMap[trainingPercentage][perceptronLogistic.activationType]['test'].append(
                perceptronLogistic.calculateError(testSet, testResults, weightsLogistic,
                                                  perceptronLogistic.activationType))
            trainingSet, trainingResults = inputUtil.splitInputFromResult(trainingSet)
            resultsMap[trainingPercentage][perceptronTanh.activationType]['training'].append(
                perceptronTanh.calculateError(trainingSet, trainingResults, weightsTanh, perceptronTanh.activationType))
            resultsMap[trainingPercentage][perceptronLogistic.activationType]['training'].append(
                perceptronLogistic.calculateError(trainingSet, trainingResults, weightsLogistic,
                                                  perceptronLogistic.activationType))

    for proportion in trainingSetProportions:
        averageListLogisticTest.append(mean(resultsMap[proportion][ActivationType.SIGMOID_LOGISTIC]['test']))
        averageListLogisticTraining.append(mean(resultsMap[proportion][ActivationType.SIGMOID_LOGISTIC]['training']))
        averageListTanhTest.append(mean(resultsMap[proportion][ActivationType.SIGMOID_TANH]['test']))
        averageListTanhTraining.append(mean(resultsMap[proportion][ActivationType.SIGMOID_TANH]['training']))

    width = 0.3
    x = np.arange(len(trainingSetProportions))
    plt.xlabel("porcentaje de entrenamiento (%)", fontsize=12)
    plt.ylabel("error cuadratico medio", fontsize=12)
    plt.title("error en ambos sets (Tanh)")
    plt.bar(x, averageListTanhTest,width, color='g', label='test')
    plt.bar(x+width, averageListTanhTraining,width, color='b',label ='training')
    plt.xticks(x + width / 2, ('10', '20', '30','40','50','60','70','80'))
    plt.legend()
    plt.savefig("./plots/errorByProportionsTanh, etha= " + str(etha) + ",beta= " + str(beta) + ", epochs= " + str(
        iterations) + ".png")
    plt.xlabel("porcentaje de entrenamiento (%)", fontsize=12)
    plt.ylabel("error cuadratico medio", fontsize=12)
    plt.title("error en ambos sets (Logistica)")
    plt.bar(x, averageListLogisticTest,width, color='g')
    plt.bar(x+width, averageListLogisticTraining,width, color='b')
    plt.savefig("./plots/errorByProportionsLogistic, etha= " + str(etha) + ",beta= " + str(beta) + ", epochs= " + str(
        iterations) + ".png")


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


def getTrainingData(inputUtil, etha, beta, iterations, trainingPercentage, activationType, selectionType, averageCount):
    errorListAcumulator = []

    for i in range(averageCount):
        trainingSet, _ = inputUtil.getTrainingSetByPercentage(trainingPercentage)
        inputCopy = copy.deepcopy(inputUtil)
        perceptron = Perceptron(copy.deepcopy(trainingSet), inputCopy, inputCopy.getWeightMatrix()
                                , activationType, etha, beta)
        _, errorList, wVsT = perceptron.trainPerceptron(inputCopy.getWeightMatrix()[0]
                                                        , iterations
                                                        , selectionType)
        errorListAcumulator.append(errorList)
    if averageCount > 1:
        return errorListAcumulator
    else:
        return wVsT, perceptron


def exportXYZModel(inputUtil, etha, beta, trainingPercentage, iterations, activationType, selectionType):
    wVsT, perceptron = getTrainingData(inputUtil, etha, beta, iterations, trainingPercentage, activationType,
                                       selectionType, 1)
    
    if os.path.exists("model.xyz"):
        os.remove("model.xyz")

    f = open('model.xyz', 'w')
    inputMatrix, _ = inputUtil.splitInputFromResult(inputUtil.getInputMatrix())
    particleCount = inputMatrix.shape[0]
    for w in wVsT:
        f.write(str(particleCount) + '\n' + '\n')
        for inputVector in inputMatrix:
            h = np.dot(inputVector, w)
            result = perceptron.calculateO(h, perceptron.activationType)
            if perceptron.activationType != ActivationType.LINEAR:
                result = perceptron.deNormalize(result)
            f.write(str(inputVector[1]) + ' ' + str(inputVector[2]) + ' ' + str(inputVector[3]) + ' ' + str(
                result) + '\n')


def plotError(inputUtil, etha, beta, iterations, trainingPercentage, activationType, selectionType):
    x, average, minValues, maxValues = calculateError(
        getTrainingData(inputUtil, etha, beta, iterations, trainingPercentage
                        , activationType, selectionType, 5))
    plt.plot(x, average, label="error")
    plt.fill_between(x, minValues, maxValues, color="lightblue", label="error")
    plt.legend()
    if selectionType == SelectionType.EPOCA:
        plt.xlabel("epocas", fontsize=12)
    else:
        plt.xlabel("iteraciones", fontsize=12)

    plt.ylabel("error cuadratico medio", fontsize=12)
    if activationType == ActivationType.LINEAR:
        plt.title("evolución del error durante entrenamiento (Linear)")
        plt.savefig("./plots/errorDuringTrainingLinear, etha= " + str(etha) + ", iterations= " + str(
            iterations) + ", percentage= " + str(trainingPercentage) + "%.png")
    elif activationType == ActivationType.SIGMOID_LOGISTIC:
        plt.title("evolución del error durante entrenamiento (logistica)")
        plt.savefig(
            "./plots/errorByProportionsLogistic, etha= " + str(etha) + ",beta= " + str(beta) + ", iterations= " + str(
                iterations) + ", percentage= " + str(trainingPercentage) + "%.png")
    else:
        plt.title("evolución del error durante entrenamiento (Tanh)")
        plt.savefig(
            "./plots/errorByProportionsLogistic, etha= " + str(etha) + ",beta= " + str(beta) + ", iterations= " + str(
                iterations) + ", percentage= " + str(trainingPercentage) + "%.png")

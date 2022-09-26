from Utils import *
from Perceptron import trainPerceptron
import ActivationType

inputUtil = InputUtil('TP2-ej2-conjunto.csv')

inputMatrix = inputUtil.getInputMatrix()
weightMatrix = inputUtil.getWeightMatrix()
trainingSet = inputUtil.getTrainingSetByPercentage(60)

wVsIteration, errorVsIteration = trainPerceptron(trainingSet, weightMatrix[0], 90000, ActivationType.ActivationType.SIGMOID)

plotw(wVsIteration)

plotError(errorVsIteration)



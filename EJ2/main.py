from Utils import *

inputUtil = InputUtil('TP2-ej2-conjunto.csv')

inputMatrix = inputUtil.getInputMatrix()
resultVector = inputUtil.getResultVector()
weightMatrix = inputUtil.getWeightMatrix()

print(inputMatrix.shape[0])
print(inputMatrix.shape[1])


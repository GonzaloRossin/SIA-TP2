from Utils import *
from plots import *



inputUtil = InputUtil('TP2-ej2-conjunto.csv')
etha, beta, training_percentage, iterations, activationType, selectionType = getparamethers()
#plotBestProportion(inputUtil, etha, beta, iterations)
exportXYZModel(inputUtil, etha, beta, training_percentage, iterations, activationType, selectionType)
#plotError(inputUtil, etha, beta, iterations, training_percentage, activationType, selectionType)

from Utils import *
from plots import *



inputUtil = InputUtil('TP2-ej2-conjunto.csv')
etha, beta, training_percentage, iterations, activationType, selectionType, graph_option = getparamethers()
if graph_option == 1:
    plotBestProportion(inputUtil, etha, beta, iterations)
elif graph_option == 2:
    plotError(inputUtil, etha, beta, iterations, training_percentage, activationType, selectionType)
elif graph_option == 3:    
    exportXYZModel(inputUtil, etha, beta, training_percentage, iterations, activationType, selectionType)
else:
    print('invalid graph option, valid options are from 1 to 3')

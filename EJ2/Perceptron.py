def calculateError(input_vector, result_vector, weight_vector, perceptronType):
    if perceptronType == 0:
        toRet = 0
        for i in range(len(input_vector)):
            result = result_vector[i]
            toRet += (result - calculateO(input_vector[i], weight_vector[i], perceptronType)) ** 2
        return 0.5 * toRet
    else:
        return 1


def calculateDeltaW(inputVectorList, weight, perceptronType, result_vector):
    weird_N = 1
    Sum = 0
    delta_weight = []
    if perceptronType == 0:
        for i in range(len(inputVectorList)):
            Sum += (result_vector[i] - calculateO(inputVectorList[i], weight, perceptronType))
        for j in range(len(inputVectorList[0])):
            delta_weight.append(weird_N * Sum * inputVectorList[j])
        return delta_weight
    else:
        return 1


def calculateO(input_vector, weight_vector, perceptronType):
    if perceptronType == 0:
        returnSum = 0
        for i in range(len(input_vector)):
            returnSum += input_vector[i] * weight_vector[i]
        return returnSum
    else:
        return 1



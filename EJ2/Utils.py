import pandas as pd
import numpy as np


def readCSV(csv_filepath):
    df = pd.read_csv(csv_filepath)
    return df


class InputUtil:
    def __init__(self, csv_path):
        df = readCSV(csv_path)
        self.inputMatrix = np.zeros((len(df.x1), 3))
        self.resultVector = np.zeros((len(df.x1), 1))
        self.weightMatrix = np.zeros((1, 3))
        for i in range(len(df.x1)):
            for j in range(3):
                if j == 0:
                    self.inputMatrix[i][j] = df.x1[i]
                elif j == 1:
                    self.inputMatrix[i][j] = df.x2[i]
                else:
                    self.inputMatrix[i][j] = df.x3[i]

            self.resultVector[i] = df.y[i]

    def getResultVector(self):
        return self.resultVector

    def getInputMatrix(self):
        return self.inputMatrix

    def getWeightMatrix(self):
        return self.weightMatrix

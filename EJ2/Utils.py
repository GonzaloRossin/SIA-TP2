import pandas as pd


def readCSV(csv_filepath):
    df = pd.read_csv(csv_filepath)
    return df


class InputVector:
    def __init__(self, csv_path):
        self.inputVector = []
        self.resultVector = []
        df = readCSV(csv_path)
        for i in range(len(df.x1)):
            vector = [df.x1[i], df.x2[i], df.x3[i]]
            self.resultVector.append(df.y[i])
            self.inputVector.append(vector)

    def printVector(self):
        for i in range(len(self.inputVector)):
            print(self.inputVector[i], self.resultVector[i])

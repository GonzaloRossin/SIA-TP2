import numpy as np

def accuracy(out_true, out_pred):
    accuracy=np.sum(out_true == out_pred)/len(out_true)
    return accuracy
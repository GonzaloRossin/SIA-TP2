import numpy as np
import matplotlib.pyplot as plt

# plot borders between classes
def plot_decision_boundary(model, X, fig):
    # set min and max values, add padding
    x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
    y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
    h = 0.01
    # generate grid of points with distance h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot the contour
    fig.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
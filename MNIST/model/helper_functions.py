import numpy as np


def sigmoid(x):
    """
    compute signum of x
    param x : ndarray vector

    returns
        -->1 / (1 + np.exp(-x)) same shape as x
        sigmoid(-input)
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    compute tanh of input vector
    param x : ndarray vector

    returns
        --> (1-e^(-x))/(1+e^(-x)) same shape as x
        tanh(-input)
    """
    return np.tanh(x)


def sigmoid_backward(dA, Z):
    """
    compute (sigmoid derivative  of input vector)*previous sigma
    param Z: ndarray vector
    param dA : previous roo

    returns
        -->dA * s * (1 - s)
    """
    s = sigmoid(Z)

    return dA * s * (1 - s)


def tanh_backward(dA, Z):
    """
    compute (tanh derivative  of input vector)*previous sigma
    param Z: ndarray vector
    param dA : previous roo

    returns
        -->dA * (1-tanh(Z)**2)
    """
    return dA * (1 - np.tanh(Z) ** 2)



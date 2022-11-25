import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(10)


def sigmoid(x):
    """
    compute signum of x
    param x : ndarray vector

    returns
        -->1 / (1 + np.exp(-x)) same shape as x
    """

    return 1 / (1 + np.exp(-x))


def tanh(x):
    """

    """
    return np.tanh(x)


def sigmoid_backward(dA, Z):
    # (e^-x)/(1+e^-x)^2
    s = sigmoid(Z)
    print('Z.shape', Z.shape)

    return dA * s * (1 - s)


def tanh_backward(x):
    return 1 - np.tanh(x) ** 2


def initializePramaters(dim):
    # input hiddens
    """
    create w with shape (dim,1) and b scaler value
    10,3,2,1

    W1=(3,10)*X(5,160000)
    W2=(2,3)
    W3=(1,2)
    """
    parameters = {}
    L = len(dim)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(dim[l], dim[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((dim[l], 1))
    return parameters


# par = initializePramaters([10, 3, 2, 1])
# for l in range(1, 4):
#     print(par["W" + str(l)].shape)
#     print(par["b" + str(l)].shape)
#     print('---------------------')
#     print(len(par))


def transform(A_prev, W, b, actvaion="sigomid"):
    A = 0
    Z = W @ A_prev + b
    if actvaion == "sigmoid":
        A = sigmoid(Z)
    elif actvaion == "tanh":
        A = tanh(Z)

    cache = (A, W, b, Z)

    return A, cache


def forwardPord(X, parameters, activation="sigmoid"):
    """
    compute linear activation function
    :param W : initialized  wighted vector
    :param b : scaler value (bias)
    :param X : input vector of size (2,m) where is 2 number of columns and m numbers of rows=60
    :return:
            --> A signum (net)
    for(hidden)
    A?
    X=A?

    """

    L = len(parameters) // 2
    A_prev = X
    # print("L", L)
    # print("X.shape", X.shape)
    caches = []
    for l in range(1, L + 1):
        # print("W" + str(l) + ": ", parameters["W" + str(l)].shape)
        A, cache = transform(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation)

        A_prev = A
        caches.append(cache)

    return A, caches


def computeCost(AOutput, Y):
    """
    :param actual:  actual value
    :param predicted: signum(weighted @ input + b)
    :return:
            -->cost value
            -1/mYlog(yhat)+(1-Y)log(1-yhat)
    """
    m = Y.shape[1]
    cost = (-1 / m) * (Y @ np.log(AOutput).T + (1 - Y) @ np.log(1 - AOutput).T)
    cost = np.squeeze(cost)

    return cost


def transform_activation_backward(dA, cache, activation="sigmoid"):
    A_prev, W, b, Z = cache
    print('*&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('A_prev shape', A_prev.shape)
    print('W shape', W.shape)
    print('b shape', b.shape)
    m = A_prev.shape[1]
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    else:
        dZ = tanh_backward(dA, Z)
    print('dZ shape', dZ.shape)
    dW = (1 / m) * (dZ @ A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ
    print('dW shape', dW.shape)
    print('db shape', db.shape)
    print('dA_prev shape', dA_prev.shape)
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    return dW, db, dA_prev


def backward_prob(AL, Y, caches, activation="sigmoid"):
    """

    :param cost: Actual -Predicted
    :param X: input vector of size (2,m) where is 2 number of columns and m numbers of rows=60
    :param withBias: boolean for using Bias or not
    :return:
            --> grads dictionary of dw ,db
    dw=,db,da=W@dZ
    dw=dA_prev*driv
    """
    print('AL', AL.shape)
    print('Y.shape', Y.shape)

    grads = {}
    L = len(caches)
    print('L.shape', L)
    print('*************************input***********')

    dA = np.subtract(Y, AL)
    print('dA', dA.shape)
    current_cache = caches[L - 1]
    grads["dW" + str(L)], grads["db" + str(L)], grads["dA" + str(L - 1)] = transform_activation_backward(dA,
                                                                                                         current_cache,
                                                                                                         activation)
    print("*********************************")
    print('dW' + str(L), grads["dW" + str(L)].shape)
    print('db' + str(L), grads["db" + str(L)].shape)
    print('dA' + str(L - 1), grads["dA" + str(L - 1)].shape)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads["dW" + str(l + 1)], grads["db" + str(l + 1)], grads["dA" + str(l)] = transform_activation_backward(
            grads["dA" + str(l + 1)],
            current_cache,
            activation)
        print('dW' + str(l + 1), grads["dW" + str(l + 1)].shape)
        print('db' + str(l + 1), grads["db" + str(l + 1)].shape)
        print('dA' + str(l), grads["dA" + str(l)].shape)

    return grads


def update_parameters(parameters, grads, learning_rate):
    """

    :param w: weights, array
    :param b: bias
    :param X: input vector of size (2,m)
    :param Y: Actual value Vector
    :param numIter: epochs
    :param learning_rate: step size
    :param print_cost: flag to print cost
    :param withBias: flag to use bias
    :return:
            --> learned weights
            --> cost vector values
    """
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        print(parameters["W" + str(l + 1)].shape)
        print(grads["dW" + str(l + 1)].shape)
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def run():
    dim = [5, 4, 3, 1]
    X = np.random.randn(5, 4) * 0.01
    Y = np.ones((1, 4))
    par = initializePramaters(dim)
    # for l in range(1, len(dim)):
    # print("W :", par["W" + str(l)].shape)
    # print("b :", par["b" + str(l)].shape)
    # print('***********************************')
    AL, caches = forwardPord(X, par, "sigmoid")
    # print('A :', A)
    cost = computeCost(AL, Y)
    # print("cost ", cost)
    # print("len of caches", len(caches))

    grads = backward_prob(AL, Y, caches, "sigmoid")
    update_parameters(par, grads, 0.01)


def get_confusion_matrix(predicted, actual):
    """
    Assumes "1" is the positive class

    :param predicted: the predicted class labels
    :param actual: the actual class labels
    :return: Confusion matrix in the following order
                    Actual = 1 | Actual = 0
    Predicted = 1        TP    |  FP
                    -----------------------
    Predicted = 0       FN     | TN
    """
    tp, fp, tn, fn = (0, 0, 0, 0)
    predicted = predicted.to_numpy()
    # actual = actual.to_numpy()
    for i in range(len(predicted)):
        if predicted[i] == 1:
            if predicted[i] == actual[i]:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if predicted[i] == actual[i]:
                tn = tn + 1
            else:
                fn = fn + 1

    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0, 0] = tp
    confusion_matrix[0, 1] = fp
    confusion_matrix[1, 0] = fn
    confusion_matrix[1, 1] = tn
    return confusion_matrix


def plots(X):
    """

    :param X:  input vector of size (2,m)
    :return:
            --> 10 plots between features (f1,f2)..elc
    """
    columns = ['flipper_length_mm', 'bill_depth_mm', 'bill_length_mm', 'gender', 'body_mass_g']
    res = [(a, b) for idx, a in enumerate(columns) for b in columns[idx + 1:]]
    for i in res:
        plt.figure()
        sns.scatterplot(data=X, x=i[0], y=i[1], hue='species')

    plt.show()


def predict(X, w, b, actual):
    """

    :param X: input vector of size (2,m)
    :param w: weighted vector
    :param b: scaler ,bias
    :param actual: Actual value Vector
    :return:
            --> model accuracy :)
    """
    acc = 0
    predicted = X @ w + b
    predicted = pd.DataFrame(predicted)
    predicted = predicted[0].apply(sigmoid)
    predicted = predicted.reset_index(drop=True)
    for index in range(X.shape[0]):
        if actual[index] == predicted[index]:
            acc = acc + 1

    return (acc / X.shape[0]) * 100, get_confusion_matrix(predicted=predicted, actual=actual)


run()

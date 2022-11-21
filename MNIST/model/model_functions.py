import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(10)


def sigmoid(x):
    """
    compute signum of x

    returns
    1 -- if input greater than 0
   -1 -- if input less than 0

    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def sigmoid_backward(A):
    # (e^-x)/(1+e^-x)^2
    s = np.exp(-A) / (1 + np.exp(-A)) ** 2
    return s


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

    cache = (A, W, b)

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
    caches = []
    for l in range(1, L + 1):
        A, cache = transform(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation)
        A_prev = A
        caches.append(cache)

    return A


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


# par = initializePramaters([10, 3, 2])
# X = np.ones((10, 10))
# Y = np.ones((1, 10))
# A = forwardPord(X, par, "sigmoid")
# cost = computeCost(A, Y)
# print(cost)


def backwordProb(A, Y, caches, flag=True):
    """

    :param cost: Actual -Predicted
    :param X: input vector of size (2,m) where is 2 number of columns and m numbers of rows=60
    :param withBias: boolean for using Bias or not
    :return:
            --> grads dictionary of dw ,db
    dw=,db,da=W@dZ
    dw=dA_prev*driv
    """
    dA = np.subtract(Y, A)
    driv = 0
    if flag is True:
        driv = sigmoid_backward(A)
    elif flag is False:
        driv = tanh_backward(A)

    dZ = dA * driv
# cur_cache=caches[l] -->A,W,B
#   prev_sige*W*simoid_prob(A)



    return grads


def optimize(w, b, X, Y, numIter, learning_rate, mseThrashold, print_cost=False, withBias=False):
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
    costs = []
    m = X.shape[0]
    for i in range(0, numIter):
        for index, x in X.reset_index(drop=True).iterrows():
            x = x.to_numpy().reshape((-1, 1))

            A = forwardPord(w, b, x)
            cost = computeCost(Y[index], A)
            grads = backwordProb(x, cost, withBias)

            dw = grads["dw"]
            db = grads["db"]
            # btw if withBias boolean value equals false the returned value for db equal false
            w = w + (learning_rate * dw)
            b = b + (learning_rate * db)

        Error = np.square(Y - X @ w + b);
        mse = (1 / (2 * m)) * np.sum(Error)
        if i % 100 == 0 and print_cost is True:
            costs.append(mse)
            print("Cost after iteration %i: %f" % (i, mse))
        if (mse < mseThrashold).bool():
            print("mes ", i)
            return w, b, costs

    return w, b, costs


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000,
          learning_rate=0.01, mseThrashold=0.05, print_cost=False, withBias=False):
    """
    :param X_train: input vector of size (2,m)
    :param Y_train: Training Actual value Vector
    :param X_test: test vector of size (2,m)
    :param Y_test: Testing Actual value Vector
    :param learning_rate: step size
    :param print_cost: flag for print cost values
    :param withBias: flag for Bias
    :return:
            --> weighted vector
            --> scaler value
    """
    w, b = initializePramaters(X_train.shape[1])
    w, b, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, mseThrashold, print_cost, withBias)
    YpredTrain, _ = predict(X_train, w, b, Y_train)
    YpredTest, _ = predict(X_test, w, b, Y_test)
    print("Train accuracy :", YpredTrain)
    print("Test accuracy :", YpredTest)

    return w, b, YpredTest


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
    predicted = predicted[0].apply(signum)
    predicted = predicted.reset_index(drop=True)
    for index in range(X.shape[0]):
        if actual[index] == predicted[index]:
            acc = acc + 1

    return (acc / X.shape[0]) * 100, get_confusion_matrix(predicted=predicted, actual=actual)

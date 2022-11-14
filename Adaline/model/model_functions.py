import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def signum(x):
    """
    compute signum of x

    returns
    1 -- if input greater than 0
   -1 -- if input less than 0

    """
    if x >= 0:
        return 1
    return -1


def initializePramaters(dim):
    """
    create w with shape (dim,1) and b scaler value

    """
    w = np.random.randn(dim, 1) * 0.01
    b = 0

    return w, b


def forwardPord(W, b, X):
    """
        compute linear activation function
    :param W : initialized  wighted vector
    :param b : scaler value (bias)
    :param X : input vector of size (2,m) where is 2 number of columns and m numbers of rows=60
    :return:
            --> A signum (net)

    """

    Z = W.T @ X + b
    A = Z
    return A


def computeCost(actual, predicted):
    """

    :param actual:  actual value
    :param predicted: signum(weighted @ input + b)
    :return:
            -->cost value
    """

    cost = actual - predicted

    return cost


def backwordProb(X, cost, withBias=False):
    """

    :param cost: Actual -Predicted
    :param X: input vector of size (2,m) where is 2 number of columns and m numbers of rows=60
    :param withBias: boolean for using Bias or not
    :return:
            --> grads dictionary of dw ,db
    """

    dw = X * cost
    db = 0
    if withBias == True:
        db = cost

    grads = {"dw": dw,
             "db": db}
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
        if i % 100 == 0 and print_cost == True:
            costs.append(mse)
            print("Cost after iteration %i: %f" % (i, mse))
        if (mse < mseThrashold).bool():
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

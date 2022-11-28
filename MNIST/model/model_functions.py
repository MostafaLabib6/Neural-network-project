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


def initialize_parameters(dim):
    """
    compute weights metrix and biases vectors
    we suppose that input features concatenated with hidden layers list
    dim[10,3,2,1]--> that mean input layer has 10 features

    param dim :list of hidden layer

    returns
        --> parameters dictionary that contains our initialized weights and biases

    create w with shape (dim[l],dim[l-1]) and b vector (dim[1],1)
    10,3,2,1
  ------------------------
    W1=(3,10)
    b1=(3,10)
    W2=(2,3)
    b2=(2,1)
    W3=(1,2)
    b3=(1,1)

    """

    parameters = {}  # dictionary that contains our parameters
    L = len(dim)

    for layer_index in range(1, L):
        parameters["W" + str(layer_index)] = np.random.randn(dim[layer_index], dim[layer_index - 1]) * 0.01
        parameters["b" + str(layer_index)] = np.zeros((dim[layer_index], 1))
        print('W' + str(layer_index), parameters["W" + str(layer_index)].shape)
        print('b' + str(layer_index), parameters["b" + str(layer_index)].shape)
    print('--------------------------------')
    return parameters


def transform_activation_forward(A_prev, W, b, activation="sigmoid"):
    """
    compute sigmoid | tanh for linear activation function
    param A_prev: input for hidden layer
    param W : weights matrix for this layer
    param b : bias vector for this layer
    param activation : activation function for this layer

    returns
        -->
        A :
        cache : that contains input , weights ,bias and Z for this layer
    """

    Z = W @ A_prev + b  # linear activation function
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "tanh":
        A = tanh(Z)
    else:
        raise Exception("Please Enter activation function ")

    cache = (A_prev, W, b, Z)

    return A, cache


def forward_propagation(X, parameters, activation="sigmoid"):
    """
    implement forward propagation -->LOOP ( linear-->Activation )

    :param X : input data ndarray
    :param parameters : initialized weights and biases
    :param activation: activation function
    :return:
            --> AL : last output vector
            -->caches :last of tuples contains ( input ,weights, bias, linear activation function )


    """

    L = len(parameters) // 2
    A = X
    caches = []

    for layer_index in range(1, L + 1):
        A_prev = A
        A, cache = transform_activation_forward(A_prev, parameters["W" + str(layer_index)],
                                                parameters["b" + str(layer_index)], activation)
        caches.append(cache)

    return A, caches


def computeCost(output_A, Y):
    """
    compute between actual vector and predicted vector
    :param output_A: probability vector for output layer
    :param Y: actual value true label
    :return:
            --> cost : (-1 / m) * (Y @ np.log(output_A).T + (1 - Y) @ np.log(1 - output_A).T)

    """

    m = Y.shape[1]
    cost = (-1 / m) * (Y @ np.log(output_A).T + (1 - Y) @ np.log(1 - output_A).T)
    cost = np.squeeze(cost)  # to remove extra []

    return cost


def transform_activation_backward(dA, cache, activation="sigmoid"):
    """
    compute dW :Gradiant of cost relative for W
    ,dB : derivation of cost relative for bias
    ,A_prev : compute new input for previous layer

    param dA: input for hidden layer ( backward )
    param cache : tuple that contains ( input ,weights, bias, linear activation function )
    param activation : activation function for this layer

    returns
        -->
        dW : metrix to update weights
        db : vector to update biases
        prev_A : new input for previous layer
    """

    A_prev, W, b, Z = cache
    m = A_prev.shape[1]

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    else:
        dZ = tanh_backward(dA, Z)

    dW = (1 / m) * (dZ @ A_prev.T)  # where is dZ is previous roo * derivative of sigmoid
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # keep output dims as input dims
    dA_prev = W.T @ dZ

    return dW, db, dA_prev


def backward_propagation(AL, Y, caches, activation="sigmoid"):
    """

    :param AL: predicated vector
    :param Y: actual vector
    :param caches: list of tuples that contains ( input ,weights, bias, linear activation function )
    :param activation: activation function

    :return:
            --> grads dictionary of dw ,db,dA
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """

    grads = {}
    L = len(caches)

    dAL = np.subtract(Y, AL)

    current_cache = caches[L - 1]
    grads["dW" + str(L)], grads["db" + str(L)], grads["dA" + str(L - 1)] = transform_activation_backward(dAL,
                                                                                                         current_cache,
                                                                                                         activation)

    for layer_index in reversed(range(L - 1)):
        current_cache = caches[layer_index]

        grads["dW" + str(layer_index + 1)], grads["db" + str(layer_index + 1)], grads[
            "dA" + str(layer_index)] = transform_activation_backward(
            grads["dA" + str(layer_index + 1)],
            current_cache,
            activation)

        print('dW' + str(layer_index + 1), grads["dW" + str(layer_index + 1)].shape)
        print('db' + str(layer_index + 1), grads["db" + str(layer_index + 1)].shape)
        print('dA' + str(layer_index), grads["dA" + str(layer_index)].shape)

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
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def run():
    dim = [5, 4, 3, 1]
    X = np.random.randn(5, 4) * 0.01
    Y = np.ones((1, 4))
    par = initialize_parameters(dim)
    # for l in range(1, len(dim)):
    # print("W :", par["W" + str(l)].shape)
    # print("b :", par["b" + str(l)].shape)
    # print('***********************************')
    AL, caches = forward_propagation(X, par, "sigmoid")
    # print('A :', A)
    cost = computeCost(AL, Y)
    # print("cost ", cost)
    # print("len of caches", len(caches))

    grads = backward_propagation(AL, Y, caches, "sigmoid")
    par = update_parameters(par, grads, 0.01)

    print('parameters length', len(par))


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

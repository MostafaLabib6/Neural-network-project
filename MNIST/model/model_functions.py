import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from helper_functions import sigmoid, tanh, sigmoid_backward, tanh_backward

np.random.seed(10)


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
        parameters["W" + str(layer_index)] = np.random.randn(dim[layer_index], dim[layer_index - 1])
        parameters["b" + str(layer_index)] = np.zeros((dim[layer_index], 1))
    #     print('W' + str(layer_index), parameters["W" + str(layer_index)].shape)
    #     print('b' + str(layer_index), parameters["b" + str(layer_index)].shape)
    # print('--------------------------------')
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


def compute_cost(output_A, Y):
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

    dW = (dZ @ A_prev.T)  # where is dZ is previous roo * derivative of sigmoid
    db = np.sum(dZ, axis=1, keepdims=True)  # keep output dims as input dims
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

        # print('dW' + str(layer_index + 1), grads["dW" + str(layer_index + 1)].shape)
        # print('db' + str(layer_index + 1), grads["db" + str(layer_index + 1)].shape)
        # print('dA' + str(layer_index), grads["dA" + str(layer_index)].shape)

    return grads


def update_parameters(parameters, grads, learning_rate, bias=True):
    """

    :param parameters: dictionary that contains our initialized weights and biases
    :param grads: dictionary of dw ,db,dA
    :param learning_rate: (value) step size
    :param bias: (flag) to use bias or not
    :return:
            --> update weights and biases
    """
    L = len(parameters) // 2  # number of layers in the neural network
    update = 1
    if bias is False:
        update = 0
    for layer_index in range(L):
        parameters["W" + str(layer_index + 1)] = parameters["W" + str(layer_index + 1)] + learning_rate * grads[
            "dW" + str(layer_index + 1)]
        parameters["b" + str(layer_index + 1)] = parameters["b" + str(layer_index + 1)] + learning_rate * grads[
            "db" + str(layer_index + 1)] * update

    return parameters


def model(X, Y, dims, learning_rate=0.001, bias=True, activation='sigmoid', epochs=1000, print_cost=False):
    """

    :param X: input data
    :param Y: label data
    :param dims: (list of neurons) each element in this list refer to neurons in layer
    :param learning_rate: step size
    :param bias: flag to use bias or not
    :param activation: activation function to be used
    :param epochs: number of iterations
    :param print_cost:  flag for printing cost after number of iterations
    :return:
        -->dictionary of learning parameters
    """
    start = time.time()
    parameters = initialize_parameters(dims)
    X = pd.DataFrame(X)
    for i in range(0, epochs):
        # Forward propagation: [LINEAR --> SIGMOID].
        for index, x in X.T.reset_index(drop=True).iterrows():
            x = x.to_numpy().reshape((-1, 1))

            AL, caches = forward_propagation(x, parameters, activation)

            # cost = compute_cost(AL, Y)

            # Backward propagation.
            y = np.zeros((10, 1))
            y[Y[index]] = 1
            grads = backward_propagation(AL, y, caches, activation)
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate, bias)

        # if print_cost and i % 100 == 0:
        #     print("Cost after iteration %i: %f" % (i, cost))
    print('training time', time.time() - start)
    return parameters


def run():
    data = pd.read_csv(r'C:\Users\DELL\Documents\GitHub\Preceptron-Signum\MNIST\mnist-in-csv\mnist_test.csv')
    X = np.array(data.drop('label', axis=1).T)
    X = X.astype('float32') / 255

    Y = np.array(data['label'])
    Y = Y.reshape((-1, 1))
    Y = Y

    dim = [X.shape[0],32,16,8, 10]
    # 512, 256, 124, 10
    par = model(X, Y, dim, 0.01, True, 'tanh', 300, True)
    predict(X, par, Y, 'tanh')


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


def predict(X, parameters, actual, activation):
    """
    :param X: input vector of size (2,m)
    :param parameters : learned weights and biases
    :param actual: Actual value Vector
    :param activation: activation function

    :return:
            --> model accuracy :)
    """
    start = time.time()
    acc = 0
    predicted, _ = forward_propagation(X, parameters, activation)
    predicted = pd.DataFrame(predicted)
    predicted = predicted.reset_index(drop=True)  # 10000,10
    for index in range(X.shape[1]):
        print(index)
        print('---------')
        max_index = np.argmax(predicted[index])
        print(actual[index][0], max_index, np.max(predicted[index][0]))
        if actual[index][0] == max_index:
            acc = acc + 1
    print(acc)
    print('acc :', acc / X.shape[1])
    print('prediction time', time.time() - start)
    return (acc / X.shape[0]) * 100  # get_confusion_matrix(predicted=predicted, actual=actual)


run()

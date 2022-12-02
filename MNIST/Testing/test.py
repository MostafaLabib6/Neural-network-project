import numpy as np
import pandas as pd
from model.model_functions import predict, model

import pickle


def run():
    data = pd.read_csv(
        '..\mnist-in-csv\mnist_train.csv')
    X = np.array(data.drop('label', axis=1).T)
    X = X.astype('float32') / 255

    Y = np.array(data['label'])
    Y = Y.reshape((-1, 1))
    Y = Y

    # dim = [X.shape[0], 32, 16, 10]
    # # 512, 256, 124, 10
    # par = model(X, Y, dim, 0.01, True, 'sigmoid', 20, True)

    # datat = pd.read_csv(
    #     'mnist-in-csv\mnist_test.csv')
    # Xt = np.array(datat.drop('label', axis=1).T)
    # Xt = Xt.astype('float32') / 255
    #
    # Yt = np.array(datat['label'])
    # Yt = Yt.reshape((-1, 1))


    # Open the file in binary mode
    with open('parameters.pickle', 'rb') as file:
        parameters = pickle.load(file)


    predict(X, parameters, Y, 'Sigmoid')


run()

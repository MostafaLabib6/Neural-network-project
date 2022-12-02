import pandas as pd
import numpy as np


class DataPreProcessingService:
    def __init__(self):
        self.train_data = pd.read_csv(
            'mnist-in-csv\mnist_train.csv')
        self.test_data = pd.read_csv(
            'mnist-in-csv\mnist_test.csv')

    def reset(self):
        self.train_data = pd.read_csv(
            'mnist-in-csv\mnist_train.csv')
        self.test_data = pd.read_csv(
            'mnist-in-csv\mnist_test.csv')

    def Shared_preprocessing_train(self):
        np.random.seed(10)
        X = np.array(self.train_data.drop('label', axis=1).T)
        X = X.astype('float32') / 255

        Y = np.array(self.train_data['label'])
        Y = Y.reshape((-1, 1))
        return X, Y

    def Shared_preprocessing_test(self):
        np.random.seed(10)
        X = np.array(self.test_data.drop('label', axis=1).T)
        X = X.astype('float32') / 255

        Y = np.array(self.test_data['label'])
        Y = Y.reshape((-1, 1))
        return X, Y

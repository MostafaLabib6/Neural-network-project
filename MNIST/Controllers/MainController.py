import numpy as np
import model.model_functions as helper
import matplotlib.pyplot as plt
import Services.DataPreProcessingService as dpp

np.random.seed(10)


class MainController:
    def __init__(self):
        self.service = dpp.DataPreProcessingService()
        self.X_train, self.Y_train = self.service.Shared_preprocessing_train()
        self.X_test, self.Y_test = self.service.Shared_preprocessing_test()

    def reset(self):
        self.service.reset()
        self.data = self.service.Shared_preprocessing_train()

    def train_model(self, dims=[32, 16], learning_rate=0.01, bias=False, activation='Sigmoid', epochs=20):
        dims.insert(0, self.X_train.shape[0])
        dims.append(10)
        self.parameteres = helper.model(X=self.X_train, Y=self.Y_train, dims=dims, learning_rate=learning_rate,
                                        bias=bias,
                                        activation=activation, epochs=epochs, print_cost=True)
        return self.parameteres

    def test_model(self, activation='sigmoid'):
        accuracy, Cmatrx = helper.predict(self.X_test, self.parameteres, self.Y_test, activation=activation)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.matshow(Cmatrx, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(Cmatrx.shape[0]):
            for j in range(Cmatrx.shape[1]):
                ax.text(x=j, y=i, s=Cmatrx[i, j], va='center', ha='center', size='x-large')

        plt.xlabel('Prediction', fontsize=18)
        plt.ylabel('Actual', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        fig.show()

        return accuracy

    def showGraphs(self):
        self.service.reset()
        data = self.service.Shared_preprocessing_train()
        helper.plots(data)

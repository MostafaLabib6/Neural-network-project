import numpy as np
import model.model_functions as helper
import matplotlib.pyplot as plt
import seaborn as sns
import Services.DataPreProcessingService as dpp
"""
150 --50 per class  --30train  20test  --90train  -60test



"""



class MainController:

    # instance attributes
    def __init__(self):
        self.service = dpp.DataPreProcessingService()
        self.data = self.service.SharedPreProcessing()

    def reset(self):
        self.service.reset()
        self.data = self.service.SharedPreProcessing()

    # instance method
    def filter_byClass(self, class1: str, class2: str):
        self.c1 = class1
        self.c2 = class2
        self.data = self.service.classFilter(class1=class1, class2=class2)

    def filter_byFeature(self, feat1: str, feat2: str):
        self.f1 = feat1
        self.f2 = feat2
        self.y, self.ytest, self.train, self.test = self.service.FeatureFilter(feat1=feat1, feat2=feat2)

    def trainModel(self, learning_rate=0.01, bais=False, epochs=2000):
        self.w, self.b, self.acc = helper.model(X_train=self.train, learning_rate=learning_rate, withBias=bais,
                                                num_iterations=epochs, Y_train=self.y, X_test=self.test,
                                                Y_test=self.ytest,
                                                print_cost=True)
        return self.acc

    def testModel(self):
        acc, Cmatrx = helper.predict(self.test, self.w, self.b, self.ytest)
        self.test = self.test.to_numpy()
        index = np.argmin(self.test[0])
        ind = np.argmax(self.test[0])
        self.test[0][index] = -5
        self.test[0][ind] = 5
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=self.data, x=self.f1, y=self.f2, hue='species')
        plt.plot(self.test[0], ((-self.w[0] / self.w[1]) * self.test[0] - self.b / self.w[1]), color='k')
        plt.show()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.matshow(Cmatrx, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(Cmatrx.shape[0]):
            for j in range(Cmatrx.shape[1]):
                ax.text(x=j, y=i, s=Cmatrx[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Prediction', fontsize=18)
        plt.ylabel('Actual', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        fig.show()

        return acc

    def showGraphs(self):
        helper.plots(self.data)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataPreProcessingService:
    def __init__(self):
        self.data = pd.read_csv('penguins.csv')

    def reset(self):
        self.data = pd.read_csv('penguins.csv')

    def classFilter(self, class1, class2):
        if class1 == "Adelie" and class2 == "Gentoo":
            self.data = self.data.iloc[:100, :]
        if class1 == "Adelie" and class2 == "Chinstrap":
            self.data = self.data.iloc[np.r_[:50, 100:150], :]
        if class1 == "Gentoo" and class2 == "Chinstrap":
            self.data = self.data.iloc[50:, :]
        if class1 == "Gentoo" and class2 == "Adelie":
            self.data = self.data.iloc[:100, :]
        if class1 == "Chinstrap" and class2 == "Adelie":
            self.data = self.data.iloc[np.r_[:50, 100:150], :]
        if class1 == "Chinstrap" and class2 == "Gentoo":
            self.data = self.data.iloc[50:, :]
        self.data['species'] = np.where(self.data['species'] == str(class1), 1, -1)

        return self.data

    def FeatureFilter(self, feat1: str, feat2: str):
        self.data = self.data
        class1Data = self.data[:50]
        class2Data = self.data[50:100]
        trainclass1 = class1Data[:30]
        trainclass2 = class2Data[:30]
        testclass1 = class1Data[30:50]
        testclass2 = class2Data[30:50]

        train = trainclass1.append(trainclass2)
        test = testclass1.append(testclass2)
        train = train[[str(feat1), str(feat2), "species"]]
        test = test[[str(feat1), str(feat2), "species"]]
        train.sample(frac=1)

        y = train['species']
        ytest = test['species']
        train = train.drop('species', axis=1)
        test = test.drop('species', axis=1)
        y = y.to_numpy().reshape((-1, 1))
        ytest = ytest.to_numpy().reshape((-1, 1))

        return y, ytest, train, test

    def SharedPreProcessing(self):

        self.data = self.data.fillna(self.data['gender'].value_counts().index[0])
        self.data["gender"] = np.where(self.data["gender"] == "male", 1, 0)
        scaler = StandardScaler()
        self.data.iloc[:, 1:] = scaler.fit_transform(self.data.iloc[:, 1:])
        return self.data

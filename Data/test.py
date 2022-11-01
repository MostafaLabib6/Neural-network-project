import pandas as pd
import numpy as np
import sklearn.preprocessing as encode


data=pd.read_csv(r'C:\Users\mando\Downloads\Neural network project\Data\penguins.csv')
data = data.fillna(data['gender'].value_counts().index[0])
# print(data.isnull().sum())
data["gender"]= np.where(data["gender"]=="male",1,0)
# print(data.describe())
# print(data.corr())
x="gender"
data=data[[x,"bill_length_mm"]]
print(data)
data = data.apply(np.random.permutation, axis=1)  
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import model.model_functions as helper
#
# data = pd.read_csv(r'penguins.csv')
#
# data.gender.fillna(data.gender.mode()[0], inplace=True)
# data['gender'] = np.where(data['gender'] == 'male', 1, 0)
# # helper.plots(data)
# data = data[:100]
# data['species'] = np.where(data['species'] == 'Adelie', 1, -1)
#
# adelieData = data[:50]
# gentoData = data[50:100]
# trainadelie = adelieData[:30]
# traingento = gentoData[:30]
# testadelie = adelieData[30:50]
# testgento = gentoData[30:50]
#
# train = trainadelie.append(traingento)
# test = testadelie.append(testgento)
# sns.scatterplot(data=test, x='flipper_length_mm', y='bill_depth_mm', hue='species')
# train = train[['species', 'flipper_length_mm', 'bill_depth_mm']]
# test = test[['species', 'flipper_length_mm', 'bill_depth_mm']]
# train.sample(frac=1)
#
# y = train['species']
# ytest = test['species']
# train = train.drop('species', axis=1)
# test = test.drop('species', axis=1)
#
# y = y.to_numpy().reshape((-1, 1))
# ytest = ytest.to_numpy().reshape((-1, 1))
#
# w, b = helper.model(train, y, test, ytest, 600, 0.01, True, True)
#
# test = test.to_numpy()
# index = np.argmin(test[0])
# test[0][index] = b
# plt.plot(test[0], ((-w[0] / w[1]) * test[0] - b / w[1]), color='k')
# plt.show()

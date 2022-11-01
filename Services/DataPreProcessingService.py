from Data.Repository  import Repository
import Services.DependencyInjector as di

from dependency_injector.wiring import Provide, inject
import numpy as np
from dependency_injector.wiring import Provide, inject

@inject
class DataPreProcessingService:
# instance attributes
 def __init__(self  ):
        container = di.Container()
        container.wire(modules=[__name__])
        self.start()
        print("Start")
 @inject
 def start(self,repository: Repository = Provide[di.Container.repository] ):
   self._repository = repository
   
 def classFillter(self, class1,class2,data):
   if class1=="Adelie" and class2=="Gentoo":
      data=data.iloc[:100,:]
   if class1=="Adelie" and class2=="Chinstrap":
      data=data.iloc[ np.r_[:50, 100:150],:]
      # data=data.iloc[{0:50,100:150},:]
   if class1=="Gentoo" and class2=="Chinstrap":
      data=data.iloc[50:,:]
   if class1=="Gentoo" and class2=="Adelie":
      data=data.iloc[:100,:]
   if class1=="Chinstrap" and class2=="Adelie":
      data=data.iloc[ np.r_[:50, 100:150],:]
      # data=data.iloc[{0:50,100:150},:]
   if class1=="Chinstrap" and class2=="Gentoo":
      data=data.iloc[50:,:]
     
   return data
    
 def FeatureFillter(self, feat1:str,feat2:str,data):
      data=data[[str(feat1),str(feat2),"species"]]
      data = data.apply(np.random.permutation, axis=0)
      print(data)
      return data   
# instance method
 def SharedPreProcessing(self):
    self.data=self._repository.getdata()
    self.data = self.data.fillna(self.data['gender'].value_counts().index[0])
    # print(data.isnull().sum())
    self.data["gender"]= np.where(self.data["gender"]=="male",1,0)
    return self.data
    # print(data.describe())
    # print(data.corr())

 

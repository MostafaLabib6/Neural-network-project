import GUI.Controllers.DependencyInjector as di
import numpy as np
import Neural_Network.NeuralNetwork as NN
from dependency_injector.wiring import Provide, inject
import Services.DataPreProcessingService as dpp
@inject
class MainController:

# instance attributes
   def __init__(self):
        container = di.Container()
        container.wire(modules=[__name__])
        self.start()
   @inject
   def start(self, service:dpp.DataPreProcessingService= Provide[di.Container.service]):
        
        self.service = service
        self.data=service.SharedPreProcessing()

    # instance method
   def classFillter(self, class1:str,class2:str):
         self.data=self.service.classFillter(class1=class1,class2=class2,data=self.data) 
    
   def FeatureFillter(self, feat1:str,feat2:str):
     self.data=self.service.FeatureFillter(feat1=feat1,feat2=feat2,data=self.data)
     self.y=self.data["species"]
     self.data=self.data.drop("species",axis=1)
     print(self.y)
     self.y= np.where(self.y==self.y[self.y.first_valid_index()],1,-1)



   def train(self,learning_rate=0.01,bais=False,epochs=1000):
        NN.slp(data=self.data,learning_rate=learning_rate,use_bias=bais,epochs=epochs,y=self.y)
   def test(self,learning_rate=0.01,bais=False,epochs=1000):
        return""
   def showGraphs(self):
        return""
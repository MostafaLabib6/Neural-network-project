import GUI.Controllers.DependencyInjector as di

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

   def train(self,learning_rate=0.01,bais=False,epochs=1000):
        print(self.data)
        return""
   def test(self,learning_rate=0.01,bais=False,epochs=1000):
        return""
   def showGraphs(self):
        return""
import numpy as np
import pandas as pd

def signum(x):
    if x >= 0:
        return 1
    else:
        return -1
        
def slp(data, learning_rate, use_bias, epochs,y):
    if(use_bias):
        w=np.random.randn(2,1)*0.01
        b=0
    else:
         w=np.random.randn(2,1)*0.01
    for i in range(epochs):
        for index, x in data.iterrows():
        #for x in data:
           print(w.shape)
           print(x)
           ypred=signum(w.T@x)
           if y[index]!=ypred:
               Loss=y[index]-ypred
               if(use_bias):
                    w=w+learning_rate*x*Loss
                    b=b+learning_rate*Loss  
               else:
                     w=w+learning_rate*x*Loss
    return w
        

           
           
           
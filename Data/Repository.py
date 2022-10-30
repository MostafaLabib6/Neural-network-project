import pandas as pd

class Repository:


# instance method
 def getdata(self):
   return pd.read_csv('penguins.csv')


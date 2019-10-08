# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:35:15 2019

@author: Sanad
"""


import pandas as pd
import seaborn as sn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics

data=pd.read_csv(r"C:\Users\Shiva\Documents\DataSet\Dataset\Regression_analysis\diabetes.csv")
data.head()
data.columns

data.corr()
sn.heatmap(data.corr(),annot=True)
x=data.drop("Outcome",axis=1)
y=data.Outcome

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
x=pd.DataFrame(xtrain)
xx=pd.DataFrame(xtest)
y=pd.DataFrame(ytrain)
yy=pd.DataFrame(ytest)
obj=LogisticRegression()
obj.fit(x,y)
pr=obj.predict(xx)

cnf=confusion_matrix(yy,pr)


print("Accuracy:",metrics.accuracy_score(yy,pr))
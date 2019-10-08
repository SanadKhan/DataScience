# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:29:46 2019

@author: Sanad
"""

#SVM

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics 
from sklearn.grid_search import GridSearchCV


cancerdata=load_breast_cancer()

cancerdata.keys()

cancer_df=pd.DataFrame(cancerdata['data'],columns=cancerdata['feature_names'])

cancer_df.head()
cancer_df.columns


xdata=cancer_df
ydata=cancerdata['target']

Xtrain,Xtest,ytrain,ytest=train_test_split(xdata,ydata,test_size=0.3)

svm_model=SVC()

svm_model.fit(Xtrain,ytrain)

pred_cancer=svm_model.predict(Xtest)

confusion_matrix(ytest,pred_cancer)
classification_report(ytest,pred_cancer)
metrics.accuracy_score(ytest,pred_cancer)*100

param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)

grid.fit(Xtrain,ytrain)

gpredict=grid.predict(Xtest)
confusion_matrix(ytest,gpredict)
classification_report(ytest,gpredict)

metrics.accuracy_score(ytest,gpredict)*100

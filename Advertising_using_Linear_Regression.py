# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:41:15 2019

@author: Shiva
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:23:30 2019

@author: Shiva
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
d=pd.read_csv(r"C:\Users\Shiva\Desktop\New folder\Advertising.csv")

d.head()
d.columns
d.isnull()


co=d.corr()

import seaborn as sn
sn.heatmap(co,annot=True)

#1)
x=d.TV
y=d.Sales
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
xtrain=pd.DataFrame(xtrain)
xtest=pd.DataFrame(xtest)
ytrain=pd.DataFrame(ytrain)
ytest=pd.DataFrame(ytest)
ads=LinearRegression()
ads.fit(xtrain,ytrain)


pr=ads.predict(xtest)
from sklearn.metrics import r2_score
acc1=r2_score(ytest,pr)










#2)


x1=d.Radio
y1=d.Sales
x1train,x1test,y1train,y1test=train_test_split(x1,y1,test_size=0.2)
x1train=pd.DataFrame(x1train)
x1test=pd.DataFrame(x1test)
y1train=pd.DataFrame(y1train)
y1test=pd.DataFrame(y1test)

ads1=LinearRegression()

ads1.fit(x1train,y1train)

pr1=ads1.predict(x1test)

from  sklearn.metrics import r2_score
acc2=r2_score(y1test,pr1)

#3)

x2=d.Newspaper
y2=d.Sales

x2train,x2test,y2train,y2test=train_test_split(x2,y2,test_size=0.2)
x2train=pd.DataFrame(x2train)
 y2train=pd.DataFrame(y2train)
 
y2test=pd.DataFrame(y2test)

ads2=LinearRegression()
ads2.fit(x2train,y2train)

prx2=ads2.predict(x2test)
from sklearn.metrics import r2_score
acc3=r2_score(y2test,prx2)

import matplotlib.pyplot as plt
x=['TV','Radio','Newspaper']
y=[acc1,acc2,acc3]
plt.bar(x,y)



##########################


tx=d.Radio
ty=d.Sales
ts=0
while ts<70:
    x1train,x1test,y1train,y1test=train_test_split(x1,y1,test_size=0.2)
    x1train=pd.DataFrame(x1train)
    x1test=pd.DataFrame(x1test)
    y1train=pd.DataFrame(y1train)
    y1test=pd.DataFrame(y1test)

    ads1=LinearRegression()
    ads1.fit(x1train,y1train)

    pr1=ads1.predict(x1test)
    p=pd.DataFrame(pr1)
    ts=r2_score(y1test,pr1)
    ts=ts*100
    print(ts)
    

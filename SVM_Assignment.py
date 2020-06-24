# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:27:06 2020

@author: RADHIKA
"""

####################Salary data Assignment####################


import pandas as pd
import numpy as np
salarydata_train=pd.read_csv("D:\\ExcelR Data\\Assignments\\Support Vector Machines\\SalaryData_Train(1).csv")
salarydata_test=pd.read_csv("D:\\ExcelR Data\\Assignments\\Support Vector Machines\\SalaryData_Test(1).csv")


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
salarydata_train.columns
salarydata_test.columns
salarydata_train.shape
salarydata_test.shape
salarydata_train.isnull().sum
salarydata_test.isnull().sum
salarydata_train.head
salarydata_test.head
salary_columns=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in salary_columns:
    salarydata_train[i] = number.fit_transform(salarydata_train[i])
    salarydata_test[i] = number.fit_transform(salarydata_test[i])
colnames = salarydata_train.columns
len(colnames[0:13])
trainX=salarydata_train[colnames[0:13]]
trainY=salarydata_train[colnames[13]]
testX=salarydata_test[colnames[0:13]]
testY=salarydata_test[colnames[13]]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(trainX,trainY)
pred_test_linear = model_linear.predict(testX)

np.mean(pred_test_linear==testY) # Accuracy = 85.233

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(trainX,trainY)
pred_test_poly = model_poly.predict(testX)

np.mean(pred_test_poly==testY) # Accuracy = 0.7795484727755644

# kernel = rbf # radial base funciton
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(trainX,trainY)
pred_test_rbf = model_rbf.predict(testX)

np.mean(pred_test_rbf==testY) # Accuracy = 0.7964143426294821

####linear model is the best model....


#########ForestFires Assignment#####################
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import tensorflow as tf
forest=pd.read_csv("D:\\ExcelR Data\\Assignments\\Support Vector Machines\\forestfires.csv")
forest.columns
forest.head
forest.shape
forest.isnull().sum
forest.describe()
forest=forest.drop(['month','day'],axis=1)

forest.columns
###creating feature column
forest.rename(columns = {'monthapr':'apr'}, inplace = True)
forest.rename(columns = {'monthaug':'aug'}, inplace = True)
forest.rename(columns = {'monthdec':'dec'}, inplace = True)
forest.rename(columns = {'monthfeb':'feb'}, inplace = True)
forest.rename(columns = {'monthjan':'jan'}, inplace = True)
forest.rename(columns = {'monthjul':'jul'}, inplace = True)
forest.rename(columns = {'monthjun':'jun'}, inplace = True)
forest.rename(columns = {'monthmar':'mar'}, inplace = True)
forest.rename(columns = {'monthmay':'may'}, inplace = True)
forest.rename(columns = {'monthnov':'nov'}, inplace = True)
forest.rename(columns = {'monthoct':'oct'}, inplace = True)
forest.rename(columns = {'monthsep':'sep'}, inplace = True)


forest.rename(columns = {'dayfri':'fri'}, inplace = True)
forest.rename(columns = {'daymon':'mon'}, inplace = True)
forest.rename(columns = {'daysat':'sat'}, inplace = True)
forest.rename(columns = {'daysun':'sun'}, inplace = True)
forest.rename(columns = {'daythu':'thu'}, inplace = True)
forest.rename(columns = {'daytue':'tue'}, inplace = True)
forest.rename(columns = {'daywed':'wed'}, inplace = True)
forest.columns

forest['area'].tail()
forest['size_category']
forest.shape
colnames = list(forest.columns)
predictors = colnames[0:28]
target = colnames[28]

X=forest[predictors]
Y=forest[target]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , Y, test_size=0.3) 
model_linear = SVC(kernel = "linear")
model_linear.fit(X_train,y_train)
pred_test_linear = model_linear.predict(X_test)

np.mean(pred_test_linear==y_test)
from sklearn.metrics import accuracy_score,precision_score
accuracy = accuracy_score(y_test, pred_test_linear)
####Accuracy:0.9743589743589743

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(X_train,y_train)
pred_test_poly = model_poly.predict(X_test)

np.mean(pred_test_poly==y_test) # Accuracy = 0.7051282051282052

# kernel = rbf # radial base funciton
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_train,y_train)
pred_test_rbf = model_rbf.predict(X_test)

np.mean(pred_test_rbf==y_test)
####Accuracy:0.7051282051282052

#####linear model is the best model accuracy is 97%

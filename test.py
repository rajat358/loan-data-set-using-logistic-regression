# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 20:26:05 2023

@author: RAJAT
"""

import pandas as pd
train=pd.read_csv("C:/Users/RAJAT/Desktop/AE27/ML/loan_data_set.csv")
test=pd.read_csv("C:/Users/RAJAT/Desktop/AE27/ML/test.csv")

#for checking null value
train.isnull().sum()
test.isnull().sum()

#imputation perform
train["Gender"]=train["Gender"].fillna(train["Gender"].mode()[0])
train["Dependents"]=train["Dependents"].fillna(train["Dependents"].mode()[0])
train["Self_Employed"]=train["Self_Employed"].fillna(train["Self_Employed"].mode()[0])
train["LoanAmount"]=train["LoanAmount"].fillna(train["LoanAmount"].median())
train["Loan_Amount_Term"]=train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].median())
train["Credit_History"]=train["Credit_History"].fillna(train["Credit_History"].median())
train["Married"]=train["Married"].fillna(train["Married"].mode()[0])


test["Gender"]=test["Gender"].fillna(test['Gender'].mode()[0])
test["Dependents"]=test["Dependents"].fillna(test["Dependents"].mode()[0])
test["Self_Employed"]=test["Self_Employed"].fillna(test["Self_Employed"].mode()[0])
test["LoanAmount"]=test["LoanAmount"].fillna(test["LoanAmount"].median())
test["Loan_Amount_Term"]=test["Loan_Amount_Term"].fillna(test["Loan_Amount_Term"].median())
test["Credit_History"]=test["Credit_History"].fillna(test["Credit_History"].median())
test["Married"]=test["Married"].fillna(test["Married"].mode()[0])

#drop loan_id because it not needed 
train=train.drop("Loan_ID",axis=1)
test=test.drop("Loan_ID",axis=1)

#DROP USED BECAUSE IT IS CATEGORICAL TYPE 
x=train.drop("Loan_Status",axis=1)
y=train.Loan_Status

#one hot encoding
x=pd.get_dummies(x)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

#train test spliting perrform
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=(5))

#use model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#prdeiction 
pred_val=model.predict(x_test)

#find accuracy using matrix
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred_val)

#predict the test model
test["target"]=model.predict(test)
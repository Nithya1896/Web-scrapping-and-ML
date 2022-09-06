# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 21:52:47 2022

@author: Nithya Sheena
"""

import numpy as np
import pandas as pd


filePath = "C:/Users/Nithya Sheena/Downloads/"
fileName = "kepler_data.csv"
fullPath = filePath+fileName
#Loading data into data frame
kepler_df =pd.read_csv(fullPath,sep =',')

#Data Exploration :
#Display first 3 records
print("Displaying first three records")
print(kepler_df.head(3))
print("---------------------------------------------------------------------")
#Display shape of dataframe
print("Displaying shape of dataframe")
print(kepler_df.shape)
print("---------------------------------------------------------------------")
#Display info of dataframe
print("Displaying info of dataframe")
print(kepler_df.info())
print("---------------------------------------------------------------------")
#Display missing values
print("Displaying missing values")
print(kepler_df.isna().sum())
print("---------------------------------------------------------------------")

#Data Pre-processing :
#setting index:
#set column as index
kepler_df = kepler_df.set_index('kepid')    
#Dropping unnecessary columns
kepler_df=kepler_df.drop(['kepoi_name','kepler_name','koi_score',
                          'koi_teq_err1','koi_teq_err2'],axis=1)
#Separating the features from the class.
features=kepler_df.drop("koi_disposition",axis=1)
target=kepler_df["koi_disposition"] 

#encoding categorical to numerical using get_dummies
features=pd.get_dummies(features,columns=['koi_pdisposition',
                                                        'koi_tce_delivname'])
#filling missing values using simple imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
imputed_features=imputer.fit_transform(features)

#Feature scaling using min-max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)
scaled_features = scaler.transform(imputed_features)


#encoding target to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
target = le.fit_transform(target)

#Split the train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_features, target, 
                                                    test_size=0.2)

#-----------------------------------------------------------------------------
#Logistic classifier: 
from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(max_iter=1500)

#Feature Selection
#Feature selection methods are intended to reduce the number of input 
#variables to those that are believed to be most useful to a model in 
#order to predict the target variable.

#Recursive Feature Elimination for logistic regression 
from sklearn.feature_selection import RFE
rfe = RFE(log_clf,n_features_to_select = 5)
fit = rfe.fit(x_train, y_train)
print("Feature Ranking: %s" % (fit.ranking_))
selected_features_log=pd.DataFrame(scaled_features[:,[8,9,14,15,41]])
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(selected_features_log, target, test_size=0.2)

#Model Building
#training model
log_clf.fit(x_train_log,y_train_log)

#predicting the test data
y_pred_log = log_clf.predict(x_test_log)

#accuracy
from sklearn.metrics import accuracy_score
data_log = {'y_actual':y_test_log,'y_pred':y_pred_log}
pred_data_logistic=pd.DataFrame(data_log)
print(pred_data_logistic)
print(log_clf.__class__.__name__, accuracy_score(y_test_log, y_pred_log))

#evaluation metrics
#confusion matrix
from sklearn.metrics import confusion_matrix
print("confusion matrix of logistic regression:\n",confusion_matrix(y_test_log, y_pred_log))
#f1_score
from sklearn.metrics import f1_score
print("f1 score of logistic regression:\n",f1_score(y_test_log, y_pred_log,average='micro'))

#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators = 20)
#Feature Selection:
from sklearn.feature_selection import SelectFromModel
select = SelectFromModel(rnd_clf)
select.fit(x_train, y_train)
print(pd.DataFrame(x_train).columns[(select.get_support())])

selected_features_rnd=pd.DataFrame(scaled_features[:,[0, 1, 2, 9, 14, 15, 19, 26, 30, 40, 41]])
x_train_rnd, x_test_rnd, y_train_rnd, y_test_rnd = train_test_split(selected_features_rnd, target, test_size=0.2)

#Model Building
#training model
rnd_clf.fit(x_train_rnd,y_train_rnd)

#predicting the test data
y_pred_rnd = rnd_clf.predict(x_test_rnd)

#accuracy
from sklearn.metrics import accuracy_score
data_rnd = {'y_actual':y_test_rnd,'y_pred':y_pred_rnd}
pred_data_random=pd.DataFrame(data_rnd)
print(pred_data_random)
print(rnd_clf.__class__.__name__, accuracy_score(y_test_rnd, y_pred_rnd))

#confusion matrix
from sklearn.metrics import confusion_matrix
print("confusion matrix of random forest:\n",confusion_matrix(y_test_rnd, y_pred_rnd))
#f1_score
from sklearn.metrics import f1_score
print("f1_score of random forest:\n",f1_score(y_test_rnd, y_pred_rnd,average='micro'))
#-----------------------------------------------------------------------------

#Answers to sub questions:
#Why did you choose the particular algorithm?
#As is this assignment asked for classification for given data,I have chosen random forest classifier algorithm. 
#The random forest model gave an accuracy of 89.9% which is higher when compared logistic regression models.
#Random forest emphasizes more on feature selection to produce a higher level of accuracy 
#and is faster on larger datasets.

#Did you consider any other choice of algorithm?
#Yes, I tried with logistic regression algorithm because it is also a classification algorithm.

#What is the accuracy?
#The random forest model gave an accuracy of 89.9%.
#What are the different types of metrics that can be used to evaluate the model?
#I have used Confusion matrix, accuracy and f1 score evaluation metrics.
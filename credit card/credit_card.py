# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:19:53 2019

@author: 1628083
"""
# # Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset
dataset = pd.read_excel('default_of_credit_card_clients.xls')

#Matrix of features
X = dataset.iloc[:, :-1].values
#Vector of dependent variable
y = dataset.iloc[0:,-1].values

#for i in range(22):
#    n = i+1
#    print ("Column X",n)
#    #Matrix of features
#    X = dataset.iloc[0:, n:n+1].values

# # Splitting the dataset into the Training set and Test set
# 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20,random_state=0)
# y_train - dependent variable vector of the training set
# y_test - dependent variable vector of the test set

# # Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# # Creating a model and Fitting the Model to the Training set

################################################
################################################
################################################

from sklearn.neighbors import KNeighborsClassifier
# minkowski is for ecledian distance
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)


from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierRF.fit(X_train, y_train)

from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'linear', random_state = 0)
classifierSVM.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDT.fit(X_train, y_train)

from sklearn.svm import SVC
classifierRBF = SVC(kernel = 'rbf', random_state = 0)
classifierRBF.fit(X_train, y_train)

################################################
################################################
################################################

# Predict the observation for the test set
# y_pred is a vector of prediction 
y_pred_knn = classifierKNN.predict(X_test)

y_pred_rf = classifierRF.predict(X_test)

y_pred_SVM = classifierSVM.predict(X_test)

y_pred_DT = classifierDT.predict(X_test)

y_pred_RBF = classifierDT.predict(X_test)


# y_test consists the real values and y_pred consists of the predicted values

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix


# Create the cofusion matrix
cmknn = confusion_matrix(y_test, y_pred_knn)
cmrf = confusion_matrix(y_test, y_pred_rf)
cmsvc = confusion_matrix(y_test, y_pred_SVM)
cmdt = confusion_matrix(y_test, y_pred_DT)
cmdt = confusion_matrix(y_test, y_pred_RBF)

#print accuracy
print ("\n")
#print ("Column X",n)

#Accuracy Score

print ("\n")
print("KNN accuracy score ",accuracy_score(y_test,y_pred_knn))
print("\nRandom Forest accuracy score ",accuracy_score(y_test,y_pred_rf))
print("\nSVM accuracy score ",accuracy_score(y_test,y_pred_SVM))
print("\nDecision Tree accuracy score ",accuracy_score(y_test,y_pred_DT))
print("\nKernel RBF accuracy score ",accuracy_score(y_test,y_pred_RBF))


#Classification Report

print("\n\nKNN classification report \n\n",classification_report(y_test,y_pred_knn))
print("Random Forest classification report \n\n",classification_report(y_test,y_pred_rf))
print("SVM classification report \n\n",classification_report(y_test,y_pred_SVM))
print("Decision Tree classification report \n\n",classification_report(y_test,y_pred_DT))
print("Kernel RBF classification report \n\n",classification_report(y_test,y_pred_DT))

# ## Let's make an Artificial Neural Network

from keras.models import Sequential
from keras.layers import Dense
classifier_ann = Sequential()
classifier_ann.add(Dense(units = 12, kernel_initializer= 'uniform', activation='relu', input_dim=23))
classifier_ann.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
classifier_ann.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
classifier_ann.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
classifier_ann.add(Dense(units=1, kernel_initializer = 'uniform', activation='sigmoid'))
classifier_ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_ann.fit(x_train, y_train, batch_size=10, epochs=100)
# this returns probability so convert it Binary value - 1/0 or True / False
y_pred_ann = classifier_ann.predict(x_test)
y_pred_ann = (y_pred_ann>0.5)


from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
cm = confusion_matrix(y_test, y_pred_ann)
print ("\nANN evaluation\n")

print("\n ANN accuracy score ",accuracy_score(y_test,y_pred_ann))
print("\n\n ANN classification report \n\n",classification_report(y_test,y_pred_ann))
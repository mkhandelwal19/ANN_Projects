# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:07:34 2019

@author: 1628083
"""

# ### Deep Learning Libraries
# 
# Tensorflow - Opensource numerical library that runs very fast computation
# 
# Theano and Tensorflows are used for research and development purposes. You can build a deep neural network from scratch using these tecnologies
# 
# Keras wraps Theano and Tensorflow. You can build a deep learning model with a very few lines of code
# 

# conda install -c conda-forge tensorflow 
# conda install -c conda-forge keras
# conda update --all
# 

# # Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras



# # Importing the dataset


dataset = pd.read_csv('bank_customer_churn.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
dataset.head()


# # Encoding the categorical data


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Onehotencoder for Geograohy . Not required for gender
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# remove a dummy geography column
X = X[:, 1:]

# # Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# # Feature Scaling..... A must for Artificial Neural Network

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Let's make an Artificial Neural Network

# ### Importing the key modules

# Sequential module is required to initialize the neural network
from keras.models import Sequential
# Dense module is required to build the layers of the ANN
from keras.layers import Dense


# ### Initiializing the ANN

# Model for ANN
classifier = Sequential()


# ### Add different layers step by step

# ### Adding the first input layer with 11 input variables


# Rectifier function is used for the input layer
# Units - Number of nodes in the hidden layer 
# It is typically average of number of nodes in input and output layers
# uniform - initialize weights uniformly. Small numbers closer to 0 
#-->
# no. of inputs (in x) = 0 to 10=11, no. of outputs = 1(yes/no)
# therefore number of neurons  = (input+output)/2 = (11+1)/2 = 6

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# 2nd hidden layer knows the number of input variables
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# 3rd hidden layer knows the number of input variables
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# 4th hidden layer knows the number of input variables
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compile the Artificial Neural Network.. applying stochastic gradient descent on the ANN

# adam - the algorithm to use to find the optimal set of weights. 
#        stochastic gradient descent on the ANN

# loss function is used to calculate the optimal weight
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Making the prediction and evaluation the model

# Fit the ANN to the training set
# 
# Choose the number of Epochs

#batch-size = 10(10 recors at a time)

# The magic happens here
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
y_pred = classifier.predict(X_test)
# this returns probability so convert it Binary value - 1/0 or True / False
y_pred = (y_pred > .5)

y_pred


from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

# Create the cofusion matrix
cm = confusion_matrix(y_test, y_pred)

#print accuracy
print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
new_prediction


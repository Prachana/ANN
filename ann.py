#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 10:28:46 2017

@author: Rachana
"""

#part 1 - Data processing  
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#dependent variable
X = dataset.iloc[:, 3:13].values
#indepent varibale
y = dataset.iloc[:, 13].values


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#changeing  encoder on counrty
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#chaning encoder on sex

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#creating dummy varibale
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#part 2 - now lets make ANN
#import the keras libraries and packages.
import keras 
#import module 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


#initializing ANN
classifier = Sequential()

#adding the input layer and the first salyer hidden, with dropout

classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p=0.1))
#adding second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))

#adding output layer.  output_dim is one beacuse we need one output
classifier.add(Dense(output_dim = 1,init = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(p=0.1))
#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating the ANN 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1,init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,batch_size = 10, nb_epoch = 100 )
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10 , n_jobs = -1) 

mean = accuracies.mean()
variance = accuracies.std()

#improving the ANN
#dropout regulation on ann

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1,init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam','rmsprop']}

#gridsearch object
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

# fit grid search 
grid_search = grid_search.fix(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_










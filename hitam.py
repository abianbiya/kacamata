# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:20:44 2019

@author: achmadz
"""

#%% import needed package
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


#%% config
names = [
         "Decision Tree"]

classifiers = [
    
    DecisionTreeClassifier(max_depth=5)]

input_file = "data_mentah_sung.csv"

#%% baca data
df = pd.read_csv(input_file, sep=',')
le_kelamin = LabelEncoder()
le_bingkai = LabelEncoder()
# convert categorical column to numeric
df['kelamin'] = le_kelamin.fit_transform(df['kelamin'])
df['bingkai'] = le_bingkai.fit_transform(df['bingkai'])

# ambil features ==> kolom ketiga sampe kolom (terakhir - 1)
X = df.as_matrix(columns=df.columns[2:-1])
# ambil kelasnya ==> kolom terakhir
Y = df.as_matrix(columns=[df.columns[-1]])

# normalisasi
scaler_training = StandardScaler()
X[:, 0:-1] = scaler_training.fit_transform(X[:, 0:-1])

# pecah jadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1234)

#%% iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train.ravel())
    score = clf.score(X_test, y_test.ravel())
    
    print("{}, score: {:.2f}".format(name, score))
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
from sklearn import tree
import graphviz


#%% config
names = [
         "Decision Tree"]

classifiers = [
    
    DecisionTreeClassifier(max_depth=2, random_state=1234)]

input_file = "data_mentah_baca.csv"

#%% baca data
df = pd.read_csv(input_file, sep=',')
le_kelamin = LabelEncoder()
le_bingkai = LabelEncoder()

# convert categorical column to numeric
df['kelamin'] = le_kelamin.fit_transform(df['kelamin'])
df['bingkai'] = le_bingkai.fit_transform(df['bingkai'])

# ambil features ==> kolom ketiga sampe kolom (terakhir - 1)
# X = df.as_matrix(columns=df.columns[2:-1])
# ambil kelasnya ==> kolom terakhir
# Y = df.as_matrix(columns=[df.columns[-1]])

X = df.values[:, 2:-1]
Y = df.values[:, -1]

# print(Y)
# exit()


# normalisasi
scaler_training = StandardScaler()
X[:, 0:-1] = scaler_training.fit_transform(X[:, 0:-1])

# pecah jadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1234)

# print(y_train)
# exit()

# iterate over classifiers
for name, clf in zip(names, classifiers):
    model = clf.fit(X_train, y_train.astype('int'))
    score = clf.score(X_test, y_test.astype('int'))
    
    print("{}, score: {:.2f}".format(name, score))

dot_data = tree.export_graphviz(model, out_file=None, class_names=['baca01','baca02','baca03','baca04','baca05','baca06'])
graph = graphviz.Source(dot_data)    
graph.render("dt109_baca86_maxdepth2")

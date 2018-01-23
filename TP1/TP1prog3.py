# -*- coding: utf-8 -*-
import pylab as pl
from sklearn.model_selection import train_test_split
import random 
from sklearn.datasets import load_iris
from sklearn import neighbors

irisData=load_iris()


X=irisData.data
Y=irisData.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=random.seed())

nb_voisins = 15

clf = neighbors.KNeighborsClassifier(nb_voisins)
clf.fit(X_train, Y_train)
print(clf.score(X_test,Y_test))

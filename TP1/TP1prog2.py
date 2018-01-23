# -*- coding: utf-8 -*-
import pylab as pl

from sklearn.datasets import load_iris
from sklearn import neighbors

irisData=load_iris()


X=irisData.data
Y=irisData.target

x = 2
y = 3

nb_voisins = 15

clf = neighbors.KNeighborsClassifier(nb_voisins)
clf.fit(X, Y)
print(clf.score(X,Y))

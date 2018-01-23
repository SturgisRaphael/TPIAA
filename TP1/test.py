# -*- coding: utf-8 -*-
import random
from sklearn.datasets import load_iris
irisData=load_iris()
X=irisData.data
Y=irisData.target
from sklearn import neighbors
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
kf=KFold(n_splits=3,shuffle=True)
X=[i for i in range(20)]
print(X)
for learn,test in kf.split(X):
	print("app : ",learn," test ",test)

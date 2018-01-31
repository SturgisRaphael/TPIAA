from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree
from sklearn.datasets import make_classification
import pylab as pl
from sklearn.cross_validation import train_test_split
import random # pour pouvoir utiliser un g ́en ́erateur de nombres al ́eatoires

X,Y=make_classification(n_samples=100000,n_features=20,n_informative=15,n_classes=3)
data_train,data_test,labels_train,labels_test = train_test_split(X, Y,test_size=0.3,random_state=random.seed())

for i in range(1,21,1):
	clf=tree.DecisionTreeClassifier(max_leaf_nodes=500*i)
	clf=clf.fit(data_train,labels_train)
	print("Score sur train pour max_leaf_nodes=", (500*i), " : ", clf.score(data_train,labels_train))
	print("Score sur test pour max_leaf_nodes=", (500*i), " : ", clf.score(data_test,labels_test))
	

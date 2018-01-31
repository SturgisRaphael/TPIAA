from sklearn.datasets import make_classification
import pylab as pl
X,Y=make_classification(n_samples=200000,n_features=2,n_redundant=0,n_clusters_per_class=2,n_classes=2)
pl.scatter(X[:,0],X[:,1],c=Y)
pl.show()

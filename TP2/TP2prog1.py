from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree
clf=tree.DecisionTreeClassifier(max_leaf_nodes=1)
clf=clf.fit(iris.data,iris.target)
print(clf.predict([iris.data[50,:]]))
print(clf.score(iris.data,iris.target))

tree.export_graphviz(clf, out_file="essai.dot")

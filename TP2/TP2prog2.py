from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree
clf_gini=tree.DecisionTreeClassifier()
clf_gini=clf_gini.fit(iris.data,iris.target)

clf_entropy=tree.DecisionTreeClassifier(criterion = 'entropy')
clf_entropy=clf_entropy.fit(iris.data,iris.target)

tree.export_graphviz(clf_gini, out_file="tree_Gini.dot")

tree.export_graphviz(clf_entropy, out_file="tree_Entropie.dot")

import random

import math
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix


def calc_error(clf, X_test, Y_test):
    Y_predict = clf.predict(X_test)

    confusion = confusion_matrix(Y_test, Y_predict)

    print(confusion)

    e = 0
    for i in range(len(confusion)):
        for j in range(len(confusion[i])):
            if i != j:
                e = e + confusion[i][j]
    return e / len(Y_test), confusion

digits=load_digits()

X_learn, X_test, Y_learn, Y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=random.seed())

kf=KFold(shuffle=True, n_splits=10)
scoresGini=[]
scoresEntropy=[]
for k in range(1,50):
    gini = 0
    entropie = 0

    clf_gini = tree.DecisionTreeClassifier(max_leaf_nodes=10*k)
    clf_entropy = tree.DecisionTreeClassifier(max_leaf_nodes=10*k, criterion='entropy')

    for train, test in kf.split(X_learn):
        X_train = X_learn[train]
        Y_train = Y_learn[train]

        clf_gini.fit(X_train, Y_train)
        clf_entropy.fit(X_train, Y_train)

        X_testBis = X_learn[test]
        Y_testBis = Y_learn[test]

        gini += clf_gini.score(X_testBis, Y_testBis)
        entropie += clf_entropy.score(X_testBis, Y_testBis)
    scoresGini.append(gini)
    scoresEntropy.append(entropie)

print("meilleure valeur pour k gini: ",scoresGini.index(max(scoresGini))+1)
print("meilleure valeur pour k entropy: ",scoresEntropy.index(max(scoresEntropy))+1)

k_gini = 10 * (scoresGini.index(max(scoresGini))+1)
k_entropy = 10 *(scoresEntropy.index(max(scoresEntropy))+1)

clf_gini = tree.DecisionTreeClassifier(max_leaf_nodes=k_gini)
clf_entropy = tree.DecisionTreeClassifier(max_leaf_nodes=k_entropy, criterion='entropy')

clf_gini.fit(X_learn, Y_learn)
clf_entropy.fit(X_learn, Y_learn)

print("Matrice gini:")
e_gini, confusion_gini = calc_error(clf_gini, X_test, Y_test)

I_gini = [e_gini - 1.96 * math.sqrt((e_gini * (1 - e_gini))/len(X_test)), e_gini + 1.96 * math.sqrt((e_gini * (1 - e_gini))/len(X_test))]

print("Matrice entropy:")
e_entropy, confusion_entropy = calc_error(clf_entropy, X_test, Y_test)

I_entropy = [e_entropy - 1.96 * math.sqrt((e_entropy * (1 - e_entropy))/len(X_test)), e_entropy + 1.96 * math.sqrt((e_entropy * (1 - e_entropy))/len(X_test))]

print("Gini = Erreur = ",e_gini, " Interval = ", I_gini)
print("Entropy = Erreur = ",e_entropy, " Interval = ", I_entropy)


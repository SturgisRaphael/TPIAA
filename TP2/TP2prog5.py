import math
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
from sklearn import tree



def calc_error(clf, X_test, Y_test):
    Y_predict = clf.predict(X_test)

    confusion = confusion_matrix(Y_test, Y_predict)

    e = 0
    for i in range(len(confusion)):
        for j in range(len(confusion[i])):
            if i != j:
                e = e + confusion[i][j]
    return e / len(Y_test)


if __name__ == '__main__':
    nbOutOfBound = 0

    for i in range(0, 100):
        X, Y = make_classification(n_samples=100000, n_informative=15, n_features=20, n_classes=3)
        X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.95, random_state=random.seed())

        X_app, X_test, Y_app, Y_test = train_test_split(X1, Y1, test_size=0.2, random_state=random.seed())

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_app, Y_app)

        e = calc_error(clf, X_test, Y_test)

        I = [e - 1.96 * math.sqrt((e * (1 - e))/len(X_test)), e + 1.96 * math.sqrt((e * (1 - e))/len(X_test))]

        f = calc_error(clf, X2, Y2)


        if(not(I[0] < f < I[1])):
            nbOutOfBound += 1

    print(nbOutOfBound, " are out of bound")
    print(nbOutOfBound, "% are out of bound")

import matplotlib.pyplot as plt
import numpy as np

def generateData(n):
    """
    generates a 2D linearly separable dataset with 2n samples.
    """
    X = (2*np.random.rand(2*n,2)-1)/2 - 0.5
    X[:n,1] += 1
    X[n:,0] += 1
    Y = np.ones([2*n,1])
    Y[n:] -= 2
    return X,Y

def generateData2(n):
    """
    generates a 2D linearly separable dataset with 2n samples.
    """
    X = (2 * np.random.rand(2 * n, 2) - 1) / 2 - 0.5
    X[:n, 0] += 1
    X[n:, 0] += 2
    X[:n, 1] += 0.5
    Y = np.ones([2*n,1])
    Y[n:] -= 2
    return X,Y

def visualise(X,Y):
    for i in range(len(X)):
        if(Y[i] == 1):
            plt.plot(X[i][0], X[i][1], 'ob')
        else:
            plt.plot(X[i][0], X[i][1], 'xr')
    plt.show()

def visualiseVector(X,Y,w):
    x1 = -1
    x2 = 1
    y1 = 1 * w[0]/w[1]
    y2 = -1 * w[0]/w[1]
    plt.plot([x1, x2], [y1, y2], linestyle='-')
    for i in range(len(X)):
        if(Y[i] == 1):
            plt.plot(X[i][0], X[i][1], 'ob')
        else:
            plt.plot(X[i][0], X[i][1], 'xr')
    plt.xlim(-1, 1), plt.ylim(-1, 1)
    plt.show()

def isClassified(w, X, Y):
    for i in range(len(X)):
        if (Y[i] * np.vdot(w, X[i]) <= 0):
            return False
    return True


def perceptron(X,Y):
    w = np.array([0,0])
    while not isClassified(w, X, Y):
        for i in range(len(X)):
            if Y[i] * np.vdot(w, X[i]) <= 0:
                w = w + Y[i]*X[i]
    return w

def eval(X,Y,w):
    error = 0.0
    for i in range(len(X)):
        if (Y[i] * np.vdot(w, X[i]) <= 0):
            error = error + 1.0
    return error/len(X)

"""
X_learn, Y_learn = generateData(100)
X_test, Y_test = generateData(30)
w = perceptron(X_learn, Y_learn)
visualiseVector(X_learn, Y_learn, w)
print("error = ", eval(X_test, Y_test, w))
"""

X, Y = generateData2(100)
visualise(X, Y)

def AddOne(X):
    """ X est un tableau n x d ; retourne le tableau n x (d+1) consistant à rajouter une colonne de """
    (n,d) = X.shape
    Xnew = np.zeros([n,d+1])
    Xnew[:,1:]=X
    Xnew[:,0]=np.ones(n)
    return Xnew

def perceptron2(X_old, Y):
    w = np.array([0, 0, 0])
    X = AddOne(X_old)
    print(X)
    while not isClassified(w, X, Y):
        for i in range(len(X)):
            if Y[i] * np.vdot(w, X[i]) <= 0:
                w = w + Y[i] * X[i]
    return w


def visualiseVector2(X,Y,w):
    x1 = 0
    x2 = 2
    y1 = -0 * w[1]/w[2] - w[0]/w[2]
    y2 = -2 * w[1]/w[2] - w[0]/w[2]
    plt.plot([x1, x2], [y1, y2], linestyle='-')
    for i in range(len(X)):
        if(Y[i] == 1):
            plt.plot(X[i][0], X[i][1], 'ob')
        else:
            plt.plot(X[i][0], X[i][1], 'xr')
    plt.xlim(0, 2), plt.ylim(-1, 0.5)
    plt.show()

w = perceptron2(X,Y)
print("perceptron2() = ",w)
visualiseVector2(X,Y, w)
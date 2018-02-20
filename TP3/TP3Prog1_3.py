import numpy as np
import matplotlib.pyplot as plt
#Question 3
Z= np.loadtxt("TP3.data2")

X = []
Y = []
for i,j in Z:
    X.append(i)
    Y.append(j)
#Question 3.1
plt.scatter(X, Y)
#Question 3.2
def poly(x, d):
    L = []
    for i in range(1, d + 1):
        L.append(x ** i)
    return L

def polyTab(X, d):
    M = []
    for e in X:
        M.append(poly(e, d))
    return M

def AddOne(X):
    """ X est un tableau n x d ; retourne le tableau n x (d+1) consistant à rajouter une colonne de """
    (n,d) = X.shape
    Xnew = np.zeros([n,d+1])
    Xnew[:,1:]=X
    Xnew[:,0]=np.ones(n)
    return Xnew

def RegLin(X,Y):
    """ X est un tableau n x d ; Y est un tableau de dimension n x 1
    retourne le vecteur w de dimension d+1, résultat de la régression linéaire """
    Z = AddOne(X)
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Z),Z)),np.transpose(Z)),Y[:,0])

#Question 3.3

def calcPoly(V, x):
    result = 0
    for i in range(len(V)):
        result += V[i] * (x ** i)
    return result

def calcPolyTab(V, X):
    result = []
    for x in X:
        result.append(calcPoly(V, x))
    return result

for d in range(1, 11):
    A = np.array(polyTab(np.array(X), d))
    y = np.array(polyTab(Y, 1))
    Out = RegLin(A, y)
    print("coef = ", Out)
    calcPolyTab(Out, X)
    plt.plot(np.linspace(-5, 5, 50), calcPolyTab(Out, np.linspace(-5, 5, 50)), label = str(d))

plt.legend()
plt.show()
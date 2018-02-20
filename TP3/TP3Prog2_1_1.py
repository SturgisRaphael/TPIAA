import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def GenData(x_min, x_max, w, nbEx, sigma):
    """ génère aléatoirement n données du type (x,w0 + <w_1:n,x> + e) où
    - w est un vecteur de dimension d + 1
    - x_min <= |x_i| <= x_max pour les d coordonnées x_i de x
    - e est un bruit Gaussien de moyenne nulle et d’écart type sigma
    Retourne deux np.array de forme (nbEx,1)
    """
    d = len(w) - 1
    X = (x_max-x_min)*np.random.rand(nbEx,d) + x_min
    Y = np.dot(X,w[1:]) + w[0] + np.random.normal(0,sigma,nbEx)
    Y = Y.reshape(nbEx,1)
    return X,Y

x_min = -5
x_max = 5
sigma = 1.0
w = np.array([-1, 2, -1, 3])
nbEx = 1000

X,Y = GenData(x_min, x_max, w,nbEx,sigma)

reg = LinearRegression()
reg.fit(X, Y)
print(reg.coef_)
print(reg.intercept_)

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

print(RegLin(X,Y))
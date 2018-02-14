import numpy as np
import matplotlib.pyplot as plt
# commandes utiles
X = np.zeros([5,3]) # tableau de 0 de dimension 5x3
Y = np.ones([3,2]) # tableau de 1 de dimension 5x3
v = np.ones(3) # vecteur contenant trois 1

X[1:4,:2] = Y # remplacement d’une partie du tableau X
X.shape # dimensions de X
np.random.rand(10) # 10 nombres aléatoires entre 0 et 1
Z = np.random.random([4,4]) # matrice aléatoire
np.random.normal(0,1,10) # 10 nombres aléatoires générés par la Gaussienne N(0,1)
np.dot(X,Y) # produit matriciel
np.dot(X,v) # produit de la matrice X et du vecteur v
np.transpose(X) # transposée de X
np.linalg.inv(Z) # inverse de Z

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

def RSS(X,Y,w):
    """ Residual Sum of Squares """
    v = Y[:,0]- (np.dot(X,w[1:]) + w[0])
    return np.dot(v,v)

x_min = -5
x_max = 5
nbEx = 10
sigma = 1.0
w = np.array([-1,2])
X,Y = GenData(x_min, x_max, w,nbEx,sigma)
print(X.shape,Y.shape)
w_estime = RegLin(X,Y)
print(RSS(X,Y,w),RSS(X,Y,w_estime))
plt.scatter(X,Y)
plt.plot([x_min,x_max],[w[1]*x_min + w[0],w[1]*x_max + w[0]], label = "Cible")
plt.plot([x_min,x_max],[w_estime[1]*x_min + w_estime[0],w_estime[1]*x_max + w_estime[0]], label = "Estimation")
plt.legend()
plt.show()

w = np.array([-1,2,-1,3])

w_mean = np.zeros(4)
for i in range(10):
    X,Y = GenData(x_min, x_max, w, nbEx, sigma)
    w_mean += RegLin(X,Y)
w_mean = w_mean/10
print(w,w_mean)

nbEx = 1000
X,Y = GenData(x_min, x_max, w, nbEx, sigma)
w_estime = RegLin(X,Y)
print(w,w_estime)
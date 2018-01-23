import numpy as np

def distance_au_centre(X):
    result = 0
    for i in X:
        max = i[0] - 0.5
        for j in i:
            if(j - 0.5 > max):
                max = j - 0.5
        result += max
    return result/len(X)

def voisin_le_plus_proche_du_centre(X):
    result = X[0];
    dist = 0;
    for i in range(len(X)):
        max = X[i][0] - 0.5
        for j in X[i]:
            if(j - 0.5 > max):
                max = j - 0.5
        if(dist > max):
            result = X[i];
            dist = max;
    return result;

for d in range(1,21):
    dist = []
    v = []
    for i in range(10):
        X = np.random.rand(100,d)
        dist.append(distance_au_centre(X))
        v.append( voisin_le_plus_proche_du_centre(X))
    print(np.mean(dist), np.mean(v))

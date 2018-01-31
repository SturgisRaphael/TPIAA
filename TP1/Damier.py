import numpy as np
from sklearn.cross_validation import train_test_split
# ou from sklearn.model_selection import train_test_split
import random # pour pouvoir utiliser un g ́en ́erateur de nombres al ́eatoires
from sklearn import neighbors

def damier(dimension, grid_size, nb_examples, noise = 0):
    data = np.random.rand(nb_examples,dimension)
    labels = np.ones(nb_examples)
    for i in range(nb_examples):
        x = data[i,:];
        for j in range(dimension):
            if int(np.floor(x[j]*grid_size)) % 2 != 0:
                labels[i]=labels[i]*(-1)
            if np.random.rand()<noise:
                labels[i]=labels[i]*(-1)
    return data, labels


for dim in [2, 10]:
    for nbcases in [2, 8]:
        for noise in [0,0.2]:
            for nbex in [1000,10000]:
                data, labels = damier(dim, nbcases, nbex, noise)
                data_train,data_test,labels_train,labels_test = train_test_split(data, labels,test_size=0.3,random_state=random.seed())
                for k in [1, 5]:
                	clf = neighbors.KNeighborsClassifier(k)
                	clf.fit(data_train, labels_train)
                	print(k, dim, nbcases, noise, nbex, " -> ", clf.score(data_test,labels_test))
                	
        

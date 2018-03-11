import numpy as np
import matplotlib.pyplot as plt

def F(t, Y):
    return np.array([-4*Y[1], Y[0]])

def euler(F, a , b, y0, n):
    h = (1.0*(b-a))/n
    T = [a]
    Y = [y0]
    t = a
    y = y0
    for i in range(n):
        y = y + h * F(t, y)
        t = t + h
        T.append(t)
        Y.append(y)
    return T, Y
"""
T,Y = euler(F, 0, 3, np.array([0, 1]), 100)


y = []
for i in Y:
    y.append(i[1])



plt.plot(T, y)
plt.show()
"""

def rectangle(x, y, t):
    plt.plot([x, x], [0, y], 'b')
    plt.plot([x+t, x+t], [0, y], 'b')
    plt.plot([x, x+t], [y, y], 'b')

rectangle(2,3, 1)
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def cost(x,y,a,b,c):
    return np.sum((y - (a * x**5 + b * x**2 + c))**2)

def dcost(x,y,a,b,c):
    interim = - 2 * (y - (a * x**5 + b * x**2 + c))
    return np.sum(interim * x**5), np.sum(interim * x**2), np.sum(interim)

def GD(x,y,a,b,c,step=0.001):
    old = 0
    new = 1
    while abs(new-old) > 0:
        da, db, dc = dcost(x,y,a,b,c)
        a -= da*step
        b -= db*step
        c -= dc*step
        old = new
        new = cost(x,y,a,b,c)
    return a, b, c

rng = np.random.RandomState(2)

x = np.linspace(-1.5,1.5,150)
y = 4 * x**5 + 2 * x**2 + 1 + rng.randn(150)

a = 0
b = 0
c = 0


a, b, c = GD(x,y,a,b,c)

z = a * x**5 + b * x**2 + c

print('The values of the constants are:\na: {}\nb: {}\nc: {}'.format(a,b,c))
plt.figure()
plt.scatter(x,y)
plt.plot(x,z)
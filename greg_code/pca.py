import numpy as np
import matplotlib.pyplot as plt

mu = 10
sig = 2

X = np.random.normal(mu,sig,502)
Y = X + np.random.normal(5,1,502)
X1 = (X - np.mean(X))/np.std(X)
Y1 = (Y - np.mean(Y))/np.std(Y)

cov = np.cov(X,Y)
eig = np.linalg.eig(cov)
print(eig)

D = np.dot(eig[1][0],eig[1][1])
print("dot",D)

origin = [0],[0]
fig, ax = plt.subplots()
U = eig[1][:,0]
V = (eig[0][1]/eig[0][0]) * eig[1][:,1]
print(U, V)
ax.quiver(*origin, U, V, color=['r','b','g'])
plt.show()
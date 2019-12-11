#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas
import matplotlib.pyplot as plt
from math import e


# In[4]:


x = np.linspace(-2, 2, 100)
y = 1 / (1 + e**x)
plt.plot(x, y)
plt.show()


# In[5]:


x = np.linspace(-2, 2, 100)
y = 1 / (1 + e**((-3)*x))
plt.plot(x, y)
plt.show()


# In[6]:


x = np.linspace(-2, 2, 100)
y = 1 / (1 + e**((x+5)))
plt.plot(x, y)
plt.show()


# In[7]:


x = np.linspace(-10, 10, 100)
y = 1 / (1 + e**(x))
plt.plot(x, y)
plt.show()


# In[8]:


x = np.linspace(-2, 2, 100)
y = 1 / (1 + e**((-3*x+5)))
plt.plot(x, y)
plt.show()


# In[10]:


import numpy as np
import matplotlib.pyplot as plt 
rng = np.random.RandomState(42) 
x = 10 * rng.rand(500) 
y=2*x-1 +rng.randn(500)


# In[11]:


def F(y, x, a, b):
    cost = np.sum((y-(a*x+b))**2)/len(y)
    return cost


# In[12]:


F(y, x, 2., -1.)


# In[13]:


def Fd(y, x, a, b):
    der1 = (y-(a*x+b))**2
    der2 = 2*(y-(a*x+b))-x
    return np.sum(der1)/len(y), np.sum(der2)/len(y)
Fd(y, x, 1., 2.)


# In[14]:


def GD(y, x, a, b, n):
    hist = []
    for i in range(n):
        c = Fd(y, x, a, b)
        a = a - 0.01* (c[0])
        b = b = 0.01* (c[1])
        hist.append((a,b))
    return a, b, hist


# In[15]:


a,b,hist =GD(y,x,5,1, 1000)
print(a,b)


# In[16]:


fig, ax = plt.subplots()
for i in range(len(hist)):
    ax.cla()
    ym = hist


# In[17]:


fig, ax = plt.subplots()
for i in range(len(hist)):
    ax.cla()
    ym=hist[i][0]*x+hist[i][1]
    ax.plot(x,ym)
    ax.scatter(x,y)
    ax.set_title(str(hist[i]))
    # Note that using time.sleep does *not* work here!
    plt.pause(0.1)


# In[18]:


ym=a*x+b
plt.scatter(x,y)
plt.plot(x,ym)
plt.show()


# In[ ]:





# In[ ]:





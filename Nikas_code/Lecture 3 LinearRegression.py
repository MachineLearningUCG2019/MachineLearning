#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
import matplotlib.pyplot as plt
import numpy as np

poly_model = make_pipeline(PolynomialFeatures(7),
                                      LinearRegression())
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
poly_model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = poly_model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)


# In[10]:


import numpy as np 
from scipy import optimize

rng= np.random.RandomState(1)

def sinfit(data, a):
    x= np.array(data[0])
    
    y=a*np.sin(x)
    return y.ravel()

x= 10 * rng.rand(50)
y = 4*np.sin(x) + 0.1 * rng.randn(50)

popt, pcov = optimize.curve_fit(sinfit, (x,), y, p0=(2.,))

print(popt)


# In[21]:


import numpy as np 
from scipy import optimize

rng= np.random.RandomState(1)

def sinfit(data, a):
    x= np.array(data[0])
    
    y=a*np.sin(x)
    return y.ravel()

x= 10 * rng.rand(50)
y = 4*np.sin(x) + 0.1 * rng.randn(50) +10

popt, pcov = optimize.curve_fit(sinfit, (x,), y, p0=(2.,))

print(popt, pcov)


# In[24]:


s = np.random.normal(0, 5.,1000)


# In[25]:


plt.hist(s)


# In[27]:


s1 = np.random.normal(0, 5.,1000)
s2 = np.random.normal(0, 5.,1000)
plt.scatter(s1, s2)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
import matplotlib.pyplot as plt
import numpy as np


# In[68]:


poly_model = make_pipeline(PolynomialFeatures(10), #play with the Features
                                      LinearRegression())
#gives polinomial feature


# In[69]:


rng = np.random.RandomState(1)


# In[70]:


x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)


# In[71]:


poly_model.fit(x[:, np.newaxis], y) 
#minimize the function -> derivate all the parameters and use gradient descent to find the minimum


# In[72]:


xfit = np.linspace(0, 10, 1000)
yfit = poly_model.predict(xfit[:, np.newaxis])


# In[73]:


plt.scatter(x, y) #Just_scatter


# In[74]:


plt.plot(xfit, yfit) #Line


# In[75]:


plt.scatter(x, y)
plt.plot(xfit, yfit)
#Both


# In[76]:


import numpy as np
from scipy import optimize
from scipy.stats import norm


# In[77]:


rng = np.random.RandomState(10000)


# In[78]:


def sinfit(data, a): #model
  x = np.array(data[0])

  y = a*np.sin(x) #taking a Sin of the data and multiplied by factor
  return y.ravel() #putting a flat array


# In[79]:


x = 10 * rng.rand(50)
y = 4*np.sin(x) + 0.1 * rng.randn(50) 


# In[80]:


popt, pcov = optimize.curve_fit(sinfit, (x,), y, p0=(2.,)) #minimilization - sinfit
#x is model, y is data
#popt is the value of the parameter - slope & intercept
#pcov is p value
print(popt)


# In[81]:


#Exercise, adding a new factor 'b'
import numpy as np
from scipy import optimize
from scipy.stats import norm

rng = np.random.RandomState(1)

def sinfit(data, a, b):
  x = np.array(data[0])

  y = a*np.sin(x) + b
  return y.ravel()

x = 10 * rng.rand(50)
y = 4*np.sin(x) + 0.1 * rng.randn(50) +10

popt, pcov = optimize.curve_fit(sinfit, (x,), y, p0=(2.,1))

print(popt, pcov)
    


# In[82]:


#s = np.random.normal(mu, sigma, 1000)
s = np.random.normal(0, 5, 1000)


# In[83]:


plt.hist(s)


# In[84]:


s1 = np.random.normal(0., 5., 10000)
s2 = np.random.normal(0., 5., 10000)
plt.scatter(s1,s2)


# In[ ]:





# In[ ]:





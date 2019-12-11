#!/usr/bin/env python
# coding: utf-8

# In[1]:


dir(numpy)


# In[2]:


import numpy as np


# In[6]:


import pandas
dir(pandas)


# In[7]:


rng = np.random.RandomState(1)
rng.rand(5)


# In[8]:


rng = np.random.RandomState(1)
x = 10 * rng.rand(50)


# In[9]:


rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
print(x)


# In[10]:


y = 2 * x - 5 + rng.randn(50)


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.scatter(x, y);


# In[14]:


y = 2 * x - 5 + rng.randn(0)
plt.scatter(x, y);


# In[15]:


y = 2 * x - 5
plt.scatter(x, y);


# In[16]:


plt.plot(x, y);


# In[18]:


y = 2 * x - 5 + rng.randn(50)
plt.plot(x, y);


# In[19]:


fromsklearn.linear_modelimport LinearRegression


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


model = LinearRegression(fit_intercept=True)


# In[22]:


model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
print("Model slope: ", model.coef_[0])
print("Model intercept:", model.intercept_)


# In[ ]:





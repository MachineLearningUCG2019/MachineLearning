#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset= pd.read_csv('/Users/MacroCosmos/Desktop/GG/petrol_consumption.csv')


# In[4]:


dataset.head()


# In[12]:


x = dataset[[ "Petrol_tax"]]
y = dataset["Petrol_Consumption"]
plt.scatter(x,y)


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[15]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit (x_train, y_train)


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


model = LinearRegression(fit_intercept=True)


# In[10]:


model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
print("Model slope: ", model.coef_[0])
print("Model intercept:", model.intercept_)


# In[16]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit (x_train, y_train)


# In[17]:


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient’])


# In[22]:


coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=["Coefficient"])
coeff_df


# In[25]:


y_pred = regressor.predict(x_test)
y_pred


# In[27]:


df = pd.DataFrame({"Actual":y_test, "Predicted": y_pred})
df


# In[ ]:





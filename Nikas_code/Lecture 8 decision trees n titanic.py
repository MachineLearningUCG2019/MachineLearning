#!/usr/bin/env python
# coding: utf-8

# In[35]:


#import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = pd.read_csv('C:/Users/MacroCosmos/Desktop/GG/titanic.csv', encoding="Latin-1")
f = open('C:/Users/MacroCosmos/Desktop/GG/titanic.csv', encoding="Latin-1")


# In[36]:


f=f.readlines()
print(f)


# In[15]:


y = data['Survived']
x = data['Sex']
print(x[1])


# In[50]:


TPositive = 0
TNegative = 0
FPositive = 0 
FNegative = 0

for i in f:
    try:
        t = i.split(',')
        if t[1] == "1" and t[5] == "female":
            TPositive += 1
        if t[1] == "0" and t[5] == "male":
            TNegative += 1
        if t[1] == "1" and t[5] == "male":
            if int(t[6]) > 20:
                FPositive +=1
            if int(t[6]) < 20:
                TPositive += 1
        if t[1] == "0" and t[5] == "female":
            FNegative += 1
    except:
        pass

h = [TPositive, TNegative, FPositive, FNegative]
print(h)


# In[26]:


TRUE = h[0]+h[1]
Sum = h[0]+h[1]+h[2]+h[3]
Accuracy = TRUE / Sum
print("Accuracy is: ", Accuracy*100, '%')


# In[51]:


TRUE = h[0]+h[1]
Sum = h[0]+h[1]+h[2]+h[3]
Accuracy = TRUE / Sum
print("Accuracy is: ", Accuracy*100, '%')


# In[57]:


import matplotlib.pyplot as plt
feature1=[]
deadF=[]
for i in f:
    t= i.split(',')
    try:
        if t[1] == "0" and t[5] == "female":
            deadF.append(int(t[6]))
    except:
        pass
    
plt.hist(deadF, bins=20)
plt.show()


# In[ ]:





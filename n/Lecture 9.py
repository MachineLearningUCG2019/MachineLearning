#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
titanic = pd.read_csv('C:/Users/mailv/OneDrive/Documents/Machine_learning/titanic.csv')
import csv


# In[2]:


titanic.tail()


# In[3]:


titanic=open('C:/Users/mailv/OneDrive/Documents/Machine_learning/titanic.csv')


# In[4]:


TP=0
FN=0
FP=0
TN=0


# In[ ]:





# In[ ]:





# In[5]:


titanic=open('C:/Users/mailv/OneDrive/Documents/Machine_learning/titanic.csv')
TP=0
FN=0
FP=0
TN=0
#1=survive
for i in titanic:
    try: 
        x=i.split(',')
        if x[5]=='female':
            if x[1]=='1':
                TP+=1
            else:
                FP+=1
        if x[5]=='male':
            if x[1]=='1':
                FN+=1
            else:
                TN+=1
    except:
        pass


# In[6]:


print(TN,FN,FP,TP)


# In[7]:


Accuracy=((TN+TP)/(TN+TP+FN+FP))
print("The accuracy is:",Accuracy*100,'%')


# In[19]:


titanic=open('C:/Users/mailv/OneDrive/Documents/Machine_learning/titanic.csv')

TP=0
FN=0
FP=0
TN=0
for i in titanic:
    try: 
        x=i.split(',')
        if x[5]=='female':
            if x[1]=='1':
                TP+=1
            else:
                FP+=1
        if x[5]=='male':
            if int(x[6])<= 15:
                if x[1]=='1':
                    TP+=1
                else: 
                    FP+=1
            elif int(x[6])> 15:
                if x[1]=='1':
                    FN+=1
                else:
                    TN+=1
            else:
                if x[1]=='1':
                    FN+=1
                else:
                    TN+=1
    except:
        pass
print(TP,FN,TN,FP)


# In[20]:


Accuracy=((TN+TP)/(TN+TP+FN+FP))
print("The accuracy is:",Accuracy*100,'%')


# In[27]:


titanic=open('C:/Users/mailv/OneDrive/Documents/Machine_learning/titanic.csv')

import matplotlib.pyplot as plt
feature1=[]
deadF=[]
for i in titanic:
    t=i.split(',')
    try:
        if t[1] == "0":
            if t[5] == "female":
                deadF.append(float(t[6]))
    except:
        pass
plt.hist(deadF, bins=20)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
digits = load_digits()


# In[2]:


type(digits)


# In[3]:


dir(digits)


# In[4]:


type(digits.data)


# In[5]:


dir(digits.data)


# In[6]:


digits.data[0].shape
    


# In[7]:


import numpy as np
import matplotlib.pyplot as plt


# In[8]:


first_image = digits.data[0].reshape(8,8)
second_image = digits.data[1].reshape(8,8)


# In[9]:


first_image.shape


# In[10]:


plt.imshow(first_image, cmap=plt.cm.inferno)
plt.title("Title is: "+ str(digits.target[0]))
plt.imshow(second_image, cmap=plt.cm.inferno)
plt.title("Title is: "+ str(digits.target[1]))


# In[11]:


fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(first_image, cmap=plt.cm.inferno)
plt.title("Title is: "+ str(digits.target[0]))
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(second_image, cmap=plt.cm.inferno)
plt.title("Title is: "+ str(digits.target[1]))


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


logisticregr = LogisticRegression()
print(logisticregr)


# In[15]:


logisticregr.fit(x_train, y_train)


# In[16]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
prediction = logisticregr.predict(x_test)
print("Logistic regression accuracy: ",accuracy_score(y_test, prediction)*100, "%")


# In[23]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from
classifier = MultinomialNB()
classifier.fit(x_train, y_train)
predictionNB = classifier.predict(x_test)
print("Naive Bayes accuracy: ",accuracy_score(y_test, predictionNB)*100, "%")


# In[29]:


import pandas as pd
df = pd.DataFrame({'Actual': y_test, 'Predicted': predictionNB})
from sklearn.metrics import confusion_matrix, accuracy_score
mat = confusion_matrix(y_test, predictionNB)
print("Confusion Matrix", mat)
print("Accuracy", accuracy_score(y_test, predictionNB)*100)


# In[ ]:





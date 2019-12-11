#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


data = pd.read_csv('C:/Users/MacroCosmos/Desktop/Lecture Files/spam.csv', encoding="Latin-1")


# In[ ]:


data.head()


# In[11]:


x = data.v2
y = data.v1
x_train, x_test, y_train, y_test = train_test_split(x,y)
vect = CountVectorizer()
counts = vect.fit_transform(x_train.values)


# In[12]:


classifier = MultinomialNB()
targets = y_train.values
classifier.fit(counts, targets)


# In[14]:


counts_test = vect.transform(x_test.values)
y_pred = classifier.predict(counts_test)


# In[15]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 


# In[16]:


from sklearn.metrics import confusion_matrix, accuracy_score
mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix", mat)
print("Accuracy", accuracy_score(y_test, y_pred))


# In[ ]:





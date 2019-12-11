#!/usr/bin/env python
# coding: utf-8

# In[3]:





# In[4]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.data[0].shape
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)


# In[31]:


from sklearn import svm
clf = svm.SVC(gamma=0.0001)
clf.fit(x_train, y_train)


# In[32]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))


# In[33]:


mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix", mat)
print("Accuracy", accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:





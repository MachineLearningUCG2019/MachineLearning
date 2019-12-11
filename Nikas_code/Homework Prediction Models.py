#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('C:/Users/MacroCosmos/Desktop/Lecture Files/mushrooms.csv', encoding="Latin-1")


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB 
from sklearn import svm


# In[3]:


#Data set: Mushroom classification
data.head(10)


# In[4]:


#Cleaning up data (numerzing)
labelEncoder = preprocessing.LabelEncoder()
for col in data.columns:
    data[col] = labelEncoder.fit_transform(data[col])


# In[5]:


# Splitting test train set, with 20% of the data as the validation set
train, test = train_test_split(data, test_size=0.2, random_state=0)


# In[6]:


# Train set
train_y = train['class']
train_x = train[[x for x in train.columns if 'class' not in x]]
# Test/Validation set
test_y = test['class']
test_x = test[[x for x in test.columns if 'class' not in x]]

#Models pulled from the web
models = [SVC(kernel='rbf', random_state=0), SVC(kernel='linear', random_state=0), LogisticRegression()]
model_names = ['SVC_rbf', 'SVC_linear', 'Logistic Regression']
for i, model in enumerate(models):
    model.fit(train_x, train_y)
    print ('The accurancy of ' + model_names[i] + ' is ' + str(accuracy_score(test_y, model.predict(test_x))) )


# In[7]:


#Logistic regression based on Lectures
logisticregr = LogisticRegression()
logisticregr.fit(train_x, train_y)
predictionLR = logisticregr.predict(test_x)
print("Logistic regression accuracy: ",accuracy_score(test_y, predictionLR)*100, "%")
print(confusion_matrix(test_y, predictionLR))
print(classification_report(test_y, predictionLR))


# In[8]:


#Naive Bayes based on Lectures
gnb = GaussianNB() 
gnb.fit(train_x, train_y) 
predictionNB = gnb.predict(test_x) 
print("Naive Bayes accuracy: ",accuracy_score(test_y, predictionNB)*100, "%")
print(confusion_matrix(test_y, predictionNB))
print(classification_report(test_y, predictionNB))


# In[9]:


#Support Vector Machine based on Lectures
clf = svm.SVC(kernel='rbf') #best result with rbf
clf.fit(train_x, train_y)
predictionSVM = clf.predict(test_x)
print("Suppor Vector Machine accuracy: ",accuracy_score(test_y, predictionSVM)*100, "%")
print(confusion_matrix(test_y, predictionSVM))
print(classification_report(test_y, predictionSVM))


# In[101]:


#Radial basis function kernel produces the best prediction for this data set. The linear kernel produces 
#prediction at around 98%
#Naive bias performed worst. 


# In[ ]:





# In[ ]:





# In[ ]:





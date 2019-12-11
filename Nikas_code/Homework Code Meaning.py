#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_iris #imports iris data set, data set contains the 3 classifications of irises and their features
iris = load_iris() #assigns a irises data dictionary to a variable
x = iris.data #x are all the features of the flower
y = iris.target #y is the target set that contains the dependent variables, which are the classes of the irises

from sklearn.model_selection import train_test_split #Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) #Test size parameter decides the size of the data that has to be split as the test dataset
#Random State makes sure that every time one runs a code, same random numbers are generated as long as values are integers
from sklearn.naive_bayes import GaussianNB #This imports Naive Bayes model
gnb = GaussianNB() #Its a GaussianNB classifier that is trained using training data
gnb.fit(X_train, y_train) #This inserts parameters for training the model
y_pred = gnb.predict(X_test) #Shows the models predictions, which class it belongs to.
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
#This shows accuracy of the model in % compared to true Y values




# In[6]:


print(y)


# In[ ]:





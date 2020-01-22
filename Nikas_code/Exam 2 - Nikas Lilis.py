#!/usr/bin/env python
# coding: utf-8

# In[163]:


#1. 1)Function that calculates the entropy list
#ProbList = [[1/2, 1/2], [1/3, 1/3, 1/3]]
#def EntropySorter(ProbList):
#    from scipy.stats import entropy
#    position = 0
#    EntropyList = []
#    for PL in ProbList:
#        x = entropy(PL, base = 2)
#        EntropyList.append(x)
#    EntropyList.sort(reverse = True)
#    return EntropyList
#EntropySorter(ProbList)
    


# In[168]:


ProbList = [[1/2, 1/2], [1/3, 1/3, 1/3]] #Have to be in order
Features = ["Ft1", "Ft2"] #Have to be in order
def EntropySorter(ProbList, Features):
    from scipy.stats import entropy
    EntropyValues = []
    for PL in ProbList:
        x = entropy(PL, base = 2)
        EntropyValues.append(x)
    print(EntropyValues)
    EntropyValues.sort(reverse = True)
    EntropyFeatures = [x for _,x in sorted(zip(EntropyValues, Features))]
    return EntropyList, EntropyFeatures
EntropySorter(ProbList, Features)


# In[169]:


#1. 2)PCA for 2D data
import numpy as np
from numpy import linalg as LA
#Randomly generated data
m = 15
sd = 1
x = np.random.normal(m, sd, 50)
rng = np.random.RandomState(2)
y = x + rng.rand(50)
import matplotlib.pyplot as plt
plt.scatter (x, y)


# In[139]:


def PCA2D(x, y, sd, m):
    x = (x-m)/sd
    y = (y-m)/sd
    cov = np.cov(x,y)
    eigV, eigVec = LA.eig(cov)
    origin = [0, 0]
    eig_vec1 = eigVec[:,0]
    eig_vec2 = eigVec[:,1]
    plt.quiver(*origin, *eig_vec1, color=['r'], scale=10)
    plt.quiver(*origin, *eig_vec2, color=['b'], scale=10)
    plt.show()
    return eig_vec1, eig_vec2


# In[140]:


PCA2D(x, y, sd, m)


# In[ ]:


#2. Decision Tree Using Titanic dataset with CSV. cv =6 and cv = 8


# In[98]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
import matplotlib.pyplot as plt
data = pd.read_csv('C:/Users/MacroCosmos/Desktop/Nikas_code/DATASETS/titanic.csv', encoding="Latin-1")


# In[99]:


AgeAmmounts = data['Age'].value_counts(dropna=False)
AgeAmmountsDict = AgeAmmounts.to_dict()
AmmountList = list(AgeAmmountsDict.values())
KeysList = list(AgeAmmountsDict.keys())
KeysList = KeysList[1:]
probabilities = []
print(len(data['Age']))
for i in range(1, len(AmmountList)): #start at 1 so you skip the NaN
    probabilities.append(AmmountList[i]/(891-177))
NaNAge = np.random.choice(KeysList, 177, p=probabilities) 
#Makes a list of ages to fill in NaN, based on the data sample


# In[114]:


OldAge=list(data['Age'])
Age=[]
position=0
for i in range(len(OldAge)):
    if np.isnan(OldAge[i]) == True:
        Age.append(NaNAge[position])
        position+=1
    else: 
        Age.append(OldAge[i])
Age = pd.Series(Age)


# In[115]:


Pclass=(data['Pclass'])
Sex = pd.Series(data['Sex'].replace(['female','male'],[int(0),int(1)]))
Age = pd.Series(Age)
SibSp =(data['SibSp'])
Parch =(data['Parch'])
Fare =(data['Fare'])
features = pd.concat([Pclass, Sex, Age, SibSp, Parch, Fare], axis=1)
x = features
y = data['Survived']


# In[116]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test


# In[117]:


clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# In[118]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred) * 100, "%")


# In[119]:


from sklearn.model_selection import cross_val_score


# In[120]:


cvlist = [6, 8]
for i in cvlist:
    print("The cross-validaton method accuracy score when cv is", i, ":", cross_val_score(clf, x_train, y_train, cv = i))
#scores = cross_val_score(clf, x_train, y_train, cv = 3)
#scores


# In[ ]:





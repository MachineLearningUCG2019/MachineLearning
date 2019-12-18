#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
digits=load_digits()

from sklearn.linear_model import LogisticRegression

import sklearn

from sklearn.model_selection import train_test_split
X=digits.data
y=digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


LR=LogisticRegression()

LR.fit(X_train,y_train)
y_pred_LR=LR.predict(X_test)
print(sklearn.metrics.confusion_matrix(y_test, y_pred_LR))
print("Logistic Regression accuracy(in %):", sklearn.metrics.accuracy_score(y_test, y_pred_LR)*100)

#As you can see, Logistic Regression has an accuracy of %95,2. The confusion matrix shows that most of the results are 
#in the diagonal line which is where they are supposed to be. 


# In[2]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
y_pred_GNB = gnb.predict(X_test) 
print(sklearn.metrics.confusion_matrix(y_test,y_pred_GNB,))
print("Gaussian Naive Bayes model accuracy(in %):", sklearn.metrics.accuracy_score(y_test, y_pred_GNB)*100)

#The Gaussian Naive Bayes model produces the lowest accuracy score with %82.4. This is probably because the way this model 
#handles the data and its probabilities is suboptimal. Mainly the 16 (actual=3 pred=8) and the 14(actual=9, pred=3) are the 
#contributors as to the lower accuracy. 


# In[3]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred_svc = svclassifier.predict(X_test)
print(sklearn.metrics.confusion_matrix(y_test,y_pred_svc))
print("Support Vector Machine/Classifier accuracy(in %):", sklearn.metrics.accuracy_score(y_test, y_pred_svc)*100)
#The Support Vector Classifier has the best accuracy. It appears that drawing linear lines to classify is the best way 
#of analyzing the data in this case. 


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

rng=np.random.RandomState(40)
x = 5 * rng.rand(100)
y = 4 * (x**5) +2*(x**2) + 1 + 10*rng.rand(100) 

#Here for the noise it really depends on the input what you get for the output. I think that if you would want more
#realistic noise you would eneter something like: y = 4 * (x**5) +2*(x**2) + 1 + (10*x**4+200)*rng.rand(100) 
#Then you get a larger spread around the line. But the more you increase the spread, the more the output values of a, b and c 
#will differ. So just to show that my code works I used 10*rng.rand(100). 

def cost(y, x, a, b,c):
    cost=0.5*(np.sum(y-(a*(x**5)+b*(x**2)+c))*2/len(y))
    return cost

def derivative_values(y, x, a, b,c):
    T=y-(a*(x**5)+b*(x**2)+c)
    first_a=0.5*(np.sum((-2)*(x**5)*T))/len(y)
    second_b=0.5*(np.sum((-2)*(x**2)*T))/len(y)
    third_c=0.5*(np.sum((-2)*T))/len(y)
    return first_a, second_b, third_c
    
def GradientDescent(y,x,a,b,c,n):
    hist=[]
    for i in range(n):
        Der_Value=derivative_values(y,x,a,b,c)
        a = a - 0.00000001*Der_Value[0]
        b = b - 0.001*Der_Value[1]
        c = c - 0.01*Der_Value[2]
        hist.append((a,b,c))
    return a,b,c


plt.scatter(x,y)
plt.suptitle("The normal points")
plt.show()

a,b,c=GradientDescent(y,x,-4,10,-20,50000)
print(a,b,c)
def function(a,b,c):
    x = 5 * rng.rand(100)
    y = a * (x**5) +b*(x**2) + c  
    return x,y
    


#You have to adjust and play with the number of iterations and learning rates to see what is optimal for the situation. 
#You have to find a balance between accurcacy and speed. Additionally, the learning rates also depend on the variable.
#As you see in the derivatives, some have way bigger numbers than the others so you have to adjust the LR to have a good fit. 
#In my example here, c=6 because of the created noise. 10 multiplied by a random number from 0-1 will be about 5. So 1+5=6. 
#As mentioned before, if you have other noise the variables will also have a slightly different value. 


# In[ ]:





# In[ ]:





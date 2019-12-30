
#!/usr/bin/env python
# coding: utf-8
#grade 23/25 (the coeffeceints didn't match the input)
# In[1]:


#Problem 1
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score
digits = load_digits()


# In[2]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=0)


# In[3]:


#Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
logisticregr = LogisticRegression()
logisticregr.fit(x_train, y_train)
predictionLR = logisticregr.predict(x_test)
print("Logistic regression accuracy: ",accuracy_score(y_test, predictionLR)*100, "%")
print(confusion_matrix(y_test,predictionLR))


# In[4]:


#Naïve Bayes classifier Gaussian
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB() 
gnb.fit(x_train, y_train) 
predictionNB = gnb.predict(x_test) 
print("Naive Bayes accuracy: ",accuracy_score(y_test, predictionNB)*100, "%")
print(confusion_matrix(y_test, predictionNB))


# In[5]:


#Naive Bayes classifier Multinomial
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train, y_train)
predictionNB = classifier.predict(x_test)
print("Naive Bayes accuracy: ",accuracy_score(y_test, predictionNB)*100, "%")
print(confusion_matrix(y_test, predictionNB))


# In[6]:


#Support Vector Machines classifier
from sklearn import svm
clf = svm.SVC(kernel='linear') #most accurate result with linear compared to rbf
clf.fit(x_train, y_train)
predictionSVM = clf.predict(x_test)
print("Suppor Vector Machine accuracy: ",accuracy_score(y_test, predictionSVM)*100, "%")
print(confusion_matrix(y_test, predictionSVM))


# In[7]:


#Support Vector Machine classifier produces the best accuracy compared to all test.
#Multinomial Naive Bayes classifier proved superior compared to GaussianNB classifier but still falls behind SVM.
#When tinkering with values, Random state = 3, test size= 0.3 makes SVM and logistic regression quite close in accuracy, although SVM still leads
#With, random_state = 3, test_size= 0.3 both Naive Bayes classifiers produce very similar accuracy scores.
#All in all, Support Vector Machine classifier proves to be most relieble and produces best results even when tinkering with values.


# In[8]:


#Problem 2 
#Gradient Descent
import numpy as np
import matplotlib.pyplot as plt 
rng = np.random.RandomState(42) 
x = 5 * rng.rand(100) 
y=4*(x**5)+2*x+1 + 10*rng.randn(100)


# In[9]:


plt.scatter(x,y)


# In[10]:


def F(y, x, a, b, c):
    cost = 0.5*(np.sum((y-(a*(x**5)+b*(x**2)+c))**2)/len(y))
    return cost
initial_cost = F(y,x,10,1,1)


# In[11]:


def Fd(y, x, a, b, c):
    f = y-(a*(x**5)+b*(x**2)+c)
    fda = (-2)*f*x**5
    fdb = (-2)*f*x**2
    fdc = (-2)*f
    return np.sum(fda)/len(y), np.sum(fdb)/len(y), np.sum(fdc)/len(y)
#Fd(y, x, 4., 2., 1.)


# In[12]:


def GD(y, x, a, b, c, n):
    hist = []
    d = 0.0000001
    for i in range(n):
        z = Fd(y, x, a, b, c)
        a = a - d* (z[0])
        b = b - d* (z[1])
        c = c - d* (z[2])
        hist.append((a,b,c))
    return a, b, c


# In[13]:


a,b,c = GD(y,x,10,1,1, 10000)
print(a,b,c)


# In[17]:


ym=a*(x**5)+b
print("Initial cost = ",initial_cost)
print("Cost after Gradient Descent = ",F(y, x, a, b, c))
print("a value = ", a, "b value = ", b, "c value = ", c )
plt.scatter(x,y)
plt.plot(x,ym)
plt.show()


# In[16]:


#Alternative visualization, needs to extract hist from GD
#fig, ax = plt.subplots()
#for i in range(len(hist)):
#    ax.cla()
#    ym=hist[i][0]*x+hist[i][1]
#    ax.plot(x,ym)
#    ax.scatter(x,y)
#    ax.set_title(str(hist[i]))


# In[179]:


#Alternative gradient descent method (does not work)
#def Gradient_Descent(F,Fd,step):
#    old = 0    
#    new = 1
#    a = 0
#    b = 0
#    c = 0
#    while abs(new-old) > 0:
#        fda, fdb, fdc = Fd(y, x, a, b, c)  
#        a -= fda * step
#        b -= fda * step                
#        c -= fda * step
#        old = new                       
#        new = F(y, x, a, b, c)
#    return a, b, c, new


# In[ ]:


#a,b,c, new = Gradient_Descent(F,Fd,0.0000001)
#print(a,b,c, new)


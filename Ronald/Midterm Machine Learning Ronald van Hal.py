#Midterm Exam for Machine Learning
#Date: 18 December 2019
#Name: Ronald van Hal
#Student number: s3489906

#Imports for both questions
import numpy as np

#Imports for question 1
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 

from sklearn.metrics import confusion_matrix, accuracy_score

#Imports for question 2
import matplotlib.pyplot as plt



#Question 1:
print("\nQuestion 1:\n")
digits = load_digits() #Loading the data

#Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

#Logistic Regression
LogReg = LogisticRegression(solver='liblinear',multi_class='auto') #Make our model, the two entries are to stop future warnings
LogReg.fit(X_train,y_train) #Train our model
LogReg_predictions = LogReg.predict(X_test) #Make predictions using our model
LogReg_mat = confusion_matrix(y_test, LogReg_predictions) #Make a confusion matrix to check each of our predictions
LogReg_acc = accuracy_score(y_test, LogReg_predictions)*100 #Calculate the accuracy of our model's predictions
print("\nLogistic Regression Confusion Matrix:\n\n", LogReg_mat)
print("\nLogistic Regression Accuracy: {:.2f}%".format(LogReg_acc)) #Round our percentage to two decimal places for neatness

print("\n--------------\n")

#The other two models will be handled in a similar manner as the Logistic Regression model

#Support Vector Machines
SVC = SVC(kernel='linear')
SVC.fit(X_train, y_train)
SVC_predictions= SVC.predict(X_test)
SVC_mat = confusion_matrix(y_test,SVC_predictions)
SVC_acc = accuracy_score(y_test, SVC_predictions)*100
print("\nSupport Vector Machine Confusion Matrix:\n\n", SVC_mat)
print("\nSupport Vector Machine Accuracy: {:.2f}%".format(SVC_acc))

print("\n--------------\n")

#Gaussian Naive Bayes
GNB = GaussianNB() 
GNB.fit(X_train, y_train) 
GNB_predictions = GNB.predict(X_test) 
GNB_mat = confusion_matrix(y_test,GNB_predictions)
GNB_acc = accuracy_score(y_test, GNB_predictions)*100
print("\nGaussian Naive Bayes Confusion Matrix:\n\n", GNB_mat)
print("\nGaussian Naive Bayes Accuracy: {:.2f}%".format(GNB_acc))

print("\n--------------\n")

print('''\nLooking at the accuracy of the three models, it appears that Logistic Regression and Support Vector Machines models are the most accurate. Their accuracy is almost the same, with the SVC model being slightly better. However, both of these models are significantly better than the Gaussian Naive Bayes model. While the GNB model is not terrible, the other options are superior.''')

print("\n--------------\n")
#Question 2:
print("\nQuestion 2:\n")
#Implement our cost function
def cost(x,y,a,b,c):
    return np.sum((y - (a * x**5 + b * x**2 + c))**2)

#Make a function to calculate the derivatives of a, b and c
def dcost(x,y,a,b,c):
    interim = - 2 * (y - (a * x**5 + b * x**2 + c))
    return np.sum(interim * x**5), np.sum(interim * x**2), np.sum(interim)

#The Gradient Descent function. It has a default step size of 0.001
def GD(x,y,a,b,c,step=0.001):
    old = 0 #These two values are used to see whether there is a significant difference
    new = 1 #They are both initialised here with a difference. They will change in the first iteration to actual values instead of these dummy ones.
    while abs(new-old) > 0: #Check the difference is significant. If not, exit the loop and then the function.
        da, db, dc = dcost(x,y,a,b,c) #Calculate the different derivatives at a specific point
        a -= da*step #Change the variables using the derivatives
        b -= db*step
        c -= dc*step
        old = new #Change the old value to the current one.
        new = cost(x,y,a,b,c) #Change the current value to the next in line using the calculated derivatives.
    return a, b, c #Return the constants when no significant changes are made.

#Set a random state so that the Gradient Descent function always gives the same results. This is optional and can be commented out if need be.
rng = np.random.RandomState(2)

#Create an x and y array to be used for the Gradient Descent
x = np.linspace(-1.5,1.5,150) #Spaced out distribution, the range is almost as high as it can be. Any higher/lower limits overflows python. The amount of points is also close to its limit of overflowing.
y = 4 * x**5 + 2 * x**2 + 1 + rng.randn(150) #Create our y-values with the constants we want to have. Also add noise for every single point.

#Initial guesses for the three constants
a = 0
b = 0
c = 0

#Call the function of the Gradient Descent
a, b, c = GD(x,y,a,b,c)

#Create a fit line using the found constants and our x-values.
z = a * x**5 + b * x**2 + c

#Print our values of the constants.
print("\nThe values of the constants are:\na: {}\nb: {}\nc: {}".format(a,b,c))

#Create a plot with both x-values and y-values plus the fit line.
plt.figure()
plt.scatter(x,y)
plt.plot(x,z)
print("\n--------------\n")
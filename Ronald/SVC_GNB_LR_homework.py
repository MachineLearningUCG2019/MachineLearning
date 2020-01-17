import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/Ronald van Hal/My Documents/Python Scripts/Machine Learning/spam.csv',encoding='latin-1')

import sklearn
X = data.v2
y = data.v1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)





#x_train, x_test, y_train, y_test = train_test_split(x, y)

vect = CountVectorizer()
counts = vect.fit_transform(X_train.values).toarray()

classifier = MultinomialNB()
targets = y_train.values
classifier.fit(counts, targets)

counts_test = vect.transform(X_test.values).toarray()
y_pred = classifier.predict(counts_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix", mat)
print(classification_report(y_test,y_pred))
print("Naive Bayes model accuracy(in %)", accuracy_score(y_test, y_pred)*100)



from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(counts,y_train)
y_pred_svc = svclassifier.predict(counts_test)


print(confusion_matrix(y_test,y_pred_svc))
print(classification_report(y_test,y_pred_svc))
print('Support vector machine accuracy score:', sklearn.metrics.accuracy_score(y_test, y_pred_svc, normalize=True, sample_weight=None)*100)

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
gnb.fit(counts, targets) 
y_pred_GNB = gnb.predict(counts_test) 
print(confusion_matrix(y_test,y_pred_GNB))
print(classification_report(y_test,y_pred_GNB))
print("Gaussian Naive Bayes model accuracy(in %):", sklearn.metrics.accuracy_score(y_test, y_pred_GNB)*100)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(counts, y_train)
y_pred_LR=LR.predict(counts_test)
print(confusion_matrix(y_test,y_pred_LR))
print(classification_report(y_test,y_pred_LR))
print("Logistic Regression accuracy(in %):", sklearn.metrics.accuracy_score(y_test, y_pred_LR)*100)


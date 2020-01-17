from sklearn.datasets import load_digits

import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression(solver='liblinear',multi_class='auto')
logisticRegr.fit(X_train,y_train)
predictions = logisticRegr.predict(X_test)
#score = logisticRegr.score(x_test,y_test)
#print(score)


mat = confusion_matrix(y_test, predictions)
print("Confusion Matrix", mat)
print("Logistic Regression accuracy(in %):", sklearn.metrics.accuracy_score(y_test, predictions)*100)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred_svc = svclassifier.predict(X_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred_svc))
#print(classification_report(y_test,y_pred_svc))
print('Support vector machine accura1cy score:', sklearn.metrics.accuracy_score(y_test, y_pred_svc, normalize=True, sample_weight=None)*100)

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
y_pred_GNB = gnb.predict(X_test) 
print(confusion_matrix(y_test,y_pred_GNB))
#print(classification_report(y_test,y_pred_GNB))
print("Gaussian Naive Bayes model accuracy(in %):", sklearn.metrics.accuracy_score(y_test, y_pred_GNB)*100)
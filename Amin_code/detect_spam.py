# Naive Bayes Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv('spam.csv',encoding='latin-1')
x = data.v2
y = data.v1
x_train, x_test, y_train, y_test = train_test_split(x, y)
vect = CountVectorizer()
counts = vect.fit_transform(x_train.values)

classifier = MultinomialNB()
targets = y_train.values
classifier.fit(counts, targets)

counts_test = vect.transform(x_test.values)
y_pred = classifier.predict(counts_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 

from sklearn.metrics import confusion_matrix, accuracy_score
mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix", mat)
print("Accuracy", accuracy_score(y_test, y_pred))

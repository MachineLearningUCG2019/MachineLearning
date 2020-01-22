# Question 1 A
print("\n------------\n")
from math import log2

list_probabilities =    [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
                        [1/2, 1/3, 1/4, 1/5, 1/6, 1/7],
                        [1/2, 1/4, 1/6, 1/8, 1/10],
                        [1/2, 1/2, 1/2, 1/2, 1/2, 1/2]]
list_entropies = []

for list_entry in list_probabilities:
    entropy = 0
    for item in list_entry:
        entropy -= item * log2(item)
    list_entropies.append(entropy)

# print(list_entropies)

sorted_list_probabilities = []

max_entry_entropy = max(list_entropies)

for i in range(len(list_entropies)):
    lowest = list_entropies.index(min(list_entropies))
    sorted_list_probabilities.append(list_probabilities[lowest])
    change_lowest = list_entropies[lowest] + max_entry_entropy
    list_entropies[lowest] = change_lowest #This is to prevent from this entry appearing again in the list
    # It also prevents the amount and location of elements to be preserved

print(sorted_list_probabilities)

# This section is to check whether our new list is sorted
# new_list_entropies = []
# for list_entry in sorted_list_probabilities:
#     entropy = 0
#     for item in list_entry:
#         entropy -= item * log2(item)
#     new_list_entropies.append(entropy)

# print(new_list_entropies)


# Question 1 B

print("\n------------\n")

import numpy as np

X = np.random.normal(10,3,250)
Y = X + np.random.normal(10,2,250)


A = list(zip(X,Y))
B = np.asarray(A)

# calculate the mean of each column
M = np.mean(B.T, axis=1)
# center columns by subtracting column means
C = B - M
# calculate covariance matrix of centered matrix
V = np.cov(C.T)
# eigendecomposition of covariance matrix
values, vectors = np.linalg.eig(V)
print("\nEigenvectors:\n",vectors)
print("\nEigenvalues:\n",values)

# Question 2

print("\n-----------\n")

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score

titanic = pd.read_csv("C:/Users/Ronald/Documents/Python Scripts/Machine Learning/titanic.csv")

# The below print option was used to find any missing items
# print(titanic.isnull().sum())

def fill_in_missing(dataset,columns):
  for column in columns:
    p = dataset[column].value_counts(normalize=True)
    missing = dataset[column].isna()
    dataset.loc[missing,column] = np.random.choice(p.index, size=len(dataset[missing]),p=p.values)
  return True

fill_in_missing(titanic,["Age"])

titanic["Sex"] = titanic["Sex"].replace(['female','male'],[int(0),int(1)])

# We will be using the features Age, Sex and Pclass

features = ["Age","Sex","Pclass"]

X = titanic[features]
y = titanic["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=0)

tree = tree.DecisionTreeClassifier("entropy")
scores_cv_6 = cross_val_score(tree,X,y,cv=6)
scores_cv_8 = cross_val_score(tree,X,y,cv=8)

print("\nAccuracy scores of cv=6:\n", scores_cv_6)
print("\nAccuracy scores of cv=8:\n", scores_cv_8)
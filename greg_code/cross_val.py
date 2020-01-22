from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC

dataset = load_iris()
X = dataset.data
Y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4,random_state=0)

svc = SVC(kernel='linear',C=1)
scores = cross_val_score(svc,x_train,y_train,cv=5)
print(scores)
#svc.fit(x_train,y_train)
#y_pred = machines[m].predict(x_test)
#score = metrics.accuracy_score(y_test,y_pred)
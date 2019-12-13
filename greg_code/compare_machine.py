from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC #support vector machine
from sklearn.naive_bayes import GaussianNB  #Gauss
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.datasets import load_breast_cancer

machines = {
  "SVC":SVC(kernel="linear"),
  "Gauss":GaussianNB(),
  "LR":LogisticRegression(solver="liblinear")
}

def main():
  dataset = load_breast_cancer()
  X = dataset.data
  Y = dataset.target
  x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
  scores = {}
  for m in machines:
    print("running machine:",m)
    machines[m].fit(x_train,y_train)
    y_pred = machines[m].predict(x_test)
    score = metrics.accuracy_score(y_test,y_pred)
    scores[m] = score
    print("score:",score)
    print("confusion matrix:\n",metrics.confusion_matrix(y_test,y_pred))
    print("\n")
  return scores

main()
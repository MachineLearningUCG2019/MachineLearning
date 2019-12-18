from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC #support vector machine
from sklearn.naive_bayes import GaussianNB  #Gauss
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.datasets import load_digits

'''
  Q1:
    run function Q1 to observe the accuracy and confusion matrix for each classification method.
    From the results, Support Vector Machine has the highest overall accuracy, with an accuracy_score of 0.978.
    Logistic Regression had an accuracy_score of 0.95
    Naive Bayes was the least accurate with an accuracy_score of 0.825
    
    Q1() returns a dict, key=classification, value=accuracy_score
'''

machines = {
  "Support Vector Machine":SVC(kernel="linear"),
  "Naive Bayes":GaussianNB(),
  "Logistic Regression":LogisticRegression(solver="liblinear",multi_class="auto")
}

def Q1():
  dataset = load_digits()
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

'''
  Q2:
    run function Q2 (with plot=True to show an animated history of the gradient descent)
    Q2 also allows options for beginning values of a, b, c, and n: number of trials
    Q2() returns dict of constants a,b,c, and the final cost.
'''


import numpy as np
import matplotlib.pyplot as plt

#generate points
rng = np.random.RandomState(50)
x = np.linspace(0,2,500)
y=4*(x**5) + 2*x**2 + 1

def cost(yobs,x,a,b,c):
  return np.sum(((a*(x**5) + b*x**2 + c) - yobs)**2)

def dcost(yobs,x,a,b,c):
  dy = 2*((a*(x**5) + b*x**2 + c) - yobs)
  da = dy*(x**5)
  db = dy*(x**2)
  dc = dy
  return np.sum(da)/len(yobs), np.sum(db)/len(yobs), np.sum(dc)/len(yobs)

def gdescent(y,x,a,b,c,n,s=0.01): #n: number of trials, s: step size
  hist = []
  for i in range(n):
    da,db,dc = dcost(y,x,a,b,c)
    a -= s*da
    b -= s*db
    c -= s*dc
    hist.append([a,b,c])
  return hist

def Q2(plot=False,a=7,b=7,c=7,n=5000):
  hist = gdescent(y,x,a,b,c,n)

  consts = {
    "a":hist[-1][0],
    "b":hist[-1][1],
    "c":hist[-1][2]
  }

  error = cost(yobs=y,x=x,**consts)
  print("final result:")
  print("a:",consts["a"],"\nb:",consts["b"],"\nc:",consts["c"])
  print("cost:",error)
  if(not plot):
    return consts, error
  
  fig, ax = plt.subplots()
  for i in range(len(hist)):
    p = hist[i]
    ax.cla()
    ym = p[0]*x**5 + p[1]*x**2 + p[2]
    ax.scatter(x,y)
    ax.plot(x,ym,color="red")
    ax.set_title(str(i)+"/"+str(len(hist)))
    plt.pause(0.1)
  return consts, error

if(__name__ == "__main__"):
  while(True):
    print("enter 1 to run function Q1, 2 to run Q2, or anything else to quit")
    inp = input()
    if(inp == "1"):
      Q1()
    elif(inp == "2"):
      Q2()
    else:
      break

'''
  Greg Charitonos   S3367649    22/01/2020    Machine_Learning_Final
'''
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn import tree as sktree
from sklearn.model_selection import cross_val_score

'''
  this is the function for part 1a
  it takes a list of lists [[...], [...]], and a base (default is 2)
'''
def get_entropy(list_of_features,base=2):
  entropies = []
  for p in list_of_features:
    entropies.append(entropy(p,base=base))
  return sorted(list(zip(list_of_features,entropies)),key = lambda x: x[1])

'''
  alternative to function above, without using scipy
'''
def alt_get_entropy(list_of_features,base=2):
  entropies = []
  for p in list_of_features:
    e = -np.sum(p * np.log(p)/np.log(base))
    entropies.append(e)
  return sorted(list(zip(list_of_features,entropies)),key = lambda x: x[1])


'''
  This function uses the distribution in a DataSeries to intelligently fill in missing data 
'''
def fill_in_nan(dataset,columns):
  for c in columns:
    s = dataset[c].value_counts(normalize=True)
    missing = dataset[c].isna()
    dataset.loc[missing,c] = np.random.choice(s.index, size=len(dataset[missing]),p=s.values)
  return True


def part1a():
  LOF = [[1/6,1/6,1/6,1/6,1/6,1/6],[2/6,4/6],[3/6,1/6,1/6,1/6]]
  ANS = get_entropy(LOF,2)
  
  print(ANS)
  return True


def part1b():
  mu = 10
  sig = 2

  X = np.random.normal(mu,sig,502)
  Y = X + np.random.normal(5,1,502)
  cov = np.cov(X,Y)
  eig = np.linalg.eig(cov)
  
  values, vectors = eig
  
  print("eignenvalues:\n",values,"\n")
  print("eignenvectors:\n",vectors,"\n")
  
  show_plot = input("show plot?y/n")
  if(show_plot.lower() == "y"):
    origin = [0],[0]
    fig, ax = plt.subplots()

    ax.quiver(*origin, eig[1][:,0], eig[1][:,1], color=['r','b'])
    plt.show()
  return True

'''PART 2'''
def part2():
  titanic = pd.read_csv("../Amin_code/titanic.csv")
  features = ["Age","Sex"]
  
  #uses probabilities to fill in nans
  fill_in_nan(titanic,["Age"])

  #female -> true, male -> false
  titanic["Sex"] = titanic["Sex"].apply(lambda x: (x == "female"))

  X = titanic[features]
  Y = titanic["Survived"]
  
  tree = sktree.DecisionTreeClassifier()
  scores6 = cross_val_score(tree,X,Y,cv=6)
  scores8 = cross_val_score(tree,X,Y,cv=8)
  print("scores cv=6:\n",scores6,"\n")
  print("scores cv=8:\n",scores8,"\n")
  return True

def main():
  while True:
    print("type 1a, 1b, or 2 to select a question. type anything else to exit")
    q = input()
    if(q == "1a"):
      part1a()
    elif(q == "1b"):
      part1b()
    elif (q == "2"):
      part2()
    else:
      break
  return True

if (__name__ == "__main__"):
  main()


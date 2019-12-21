'''
  Greg Charitonos   12/21/19    kmeans_clustering_comparison
'''

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import kmeans as gregs_code #my code
import itertools


dataset = pd.read_csv("MachineLearning/Mall_Customers.csv")
cols = dataset.iloc[:,[3,4]].values

cl_num = 5 #cluster number
cl_iters = 1000 #kmeans interations

km = KMeans(
    n_clusters=cl_num, init='random',
    n_init=cl_iters, max_iter=1000
)
print("getting sklearn kmeans...")
y_km = km.fit_predict(cols)

print("getting greg's kmeans...")
Hist = gregs_code.kmeans(cl_num,cols,cl_iters,simple=True)

print("generating permutations for greg's kmeans and comparing to sklearn kmeans...")
'''
This is because the centers are placed at random, therefore center 4 for one algorithm could refer to center 2 for the other, even though they share the same points.
We loop through each permutation for greg's kmeans algorithm and compare each value with sklearn's value.
If they match up, we out "kmeans are the same!".
'''

def test_all_perms():
  perms = list(itertools.permutations([i for i in range(cl_num)]))
  for p in perms:
    trial = [p[i] for i in Hist]
    t = [trial[i] == y_km[i] for i in range(len(trial))]
    if(all(t)):
      print("kmeans are the same!")
      return True
  print("kmeans are not the same...")
  return False

test_all_perms()
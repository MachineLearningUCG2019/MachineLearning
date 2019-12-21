'''
  Greg Charitonos   12/21/19    kmeans_clustering
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

dataset = pd.read_csv("MachineLearning/Mall_Customers.csv")

cols = dataset.iloc[:,[3,4]].values
x = cols[:,0]
y = cols[:,1]
minmax_x = [min(x),max(x)]
minmax_y = [min(y),max(y)]

def dist(p1,p2):
  d = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
  return d

def randLoc(x,y):
  return x[0] + np.random.rand()*x[1], y[0] + np.random.random()*y[1]

def updateCenters(cnt,points):
  cnt = copy.deepcopy(cnt) #makes sure to create entire new dict, not just a pointer
  for c in cnt:
    cnt[c][2] = []
  for i in range(len(points)):
    D = np.inf
    C = None
    for c in cnt:
      d = dist(cnt[c],points[i])
      if(d < D): #get closest center to point
        D = d
        C = c
    cnt[C][2].append(i) #append point to center's point list
  for c in cnt:
    if(len(cnt[c][2]) > 0):
      cnt[c][0] = np.sum([points[i][0] for i in cnt[c][2]])/len(cnt[c][2])
      cnt[c][1] = np.sum([points[i][1] for i in cnt[c][2]])/len(cnt[c][2])
    else:#if for some reason the center's point list is empty, don't divide by 0, but set the center to a random position
      cnt[c] = [*randLoc(minmax_x,minmax_y),[]]
  return cnt

def kmeans(n,points):
  centers = {}
  hist = []
  for i in range(n):
    centers[i] = [*randLoc(minmax_x,minmax_y),[]] #initialise the center's position random
  hist.append(centers) #add it as our initial history
  index = 0
  while True:
    index += 1
    hist.append(updateCenters(hist[-1],points)) #add new centers to history
    ds = [] #keep track of distance between each center and its previous position
    for i in range(n):
      ds.append(dist(hist[-1][i],hist[-2][i]))
    if(sum(ds) == 0 and index > 0): #if we've had at least 1 iteration, and distance between each center = 0
      break
  return hist

Hist = kmeans(5,cols)
print("number of iterations:",len(Hist))
print("starting animation...")

fig, ax = plt.subplots()
index = 0

for p in Hist:
  ax.cla()
  ax.scatter([p[j][0] for j in p], [p[j][1] for j in p],color="red",marker="+")
  for c in p:
    plist = p[c][2]
    xs = [cols[i][0] for i in plist]
    ys = [cols[i][1] for i in plist]
    ax.scatter(xs,ys)
  ax.set_title(str(index)+" "+str(p[0][0])+" "+str(Hist[0][0][0]))
  index += 1
  plt.pause(0.3)
  if(index == len(Hist)):
    plt.pause(10)
  

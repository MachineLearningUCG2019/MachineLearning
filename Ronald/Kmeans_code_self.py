import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def distance(point_1,point_2):
    difference_x = (point_1[0] - point_2[0])**2
    difference_y = (point_1[1] - point_2[1])**2
    distance_points = (difference_x + difference_y)**0.5
    return distance_points

def centers(x_points,y_points,cluster_amount):
    centers_list = []
    random_list = []
#    cluster_limit = len(x_points)//cluster_amount
#    low_limit = 0
#    high_limit = cluster_limit
    for i in range(cluster_amount):
#        random = np.random.randint(low=low_limit,high=high_limit)
        random = np.random.randint(low=0,high=199)
        while random in random_list:
            random = np.random.randint(low=0,high=199)
        random_list.append(random)
        centers_list.append([x_points[random],y_points[random]])
#        low_limit += cluster_limit
#        high_limit += cluster_limit
    return centers_list, random_list

def allocate_points(centers_list,x_points,y_points):
    clusters = {}
    for center in centers_list:
        clusters[str(center)] = []
    for i in range(len(x_points)):
        distances = []
        point = [x_points[i],y_points[i]]
        for center in centers_list:
            distances.append(distance(center,point))
        minimum = distances.index(min(distances))
        clusters[str(centers_list[minimum])].append([x_points[i],y_points[i]])
    return clusters

def update_centers(clusters_list,centers_list):
    new_centers_list = []
    for center in centers_list:
        cluster_points = clusters_list[str(center)]
        x = 0
        y = 0
        n = 0
        for point in cluster_points:
            x += point[0]
            y += point[1]
            n += 1
        average_x = x / n
        average_y = y / n
        average_center = [average_x,average_y]
        distances = []
        for point in cluster_points:
            distances.append(distance(average_center,point))
        minimum = distances.index(min(distances))
        new_centers_list.append(cluster_points[minimum])
    return new_centers_list

def cluster_points_list(clusters_list,centers_list):
    cluster_points = []
    for center in centers_list:
        cluster_points.append(clusters_list[str(center)])
    return cluster_points

def cluster_score(clusters_list,centers_list):
    score = []
    for center in centers_list:
        distances = []
        cluster_points = clusters_list[str(center)]
        for point in cluster_points:
            distances.append(distance(center,point))
        score += [sum(distances)]
    return score

dataset = pd.read_csv('/Users/Ronald van Hal/My Documents/Python Scripts/Machine Learning/Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values

x = X[:,0]
y = X[:,1]
cluster_amount = 5

total_score = -1
loops = 0

lowest_centers_list = []
lowest_clusters_list = {}

while True:
    loops += 1
    centers_list, random_list = centers(x,y,cluster_amount)
    clusters_list = allocate_points(centers_list,x,y)
    new_centers_list = update_centers(clusters_list,centers_list)
    
    
    while centers_list != new_centers_list:
        centers_list = new_centers_list
        clusters_list = allocate_points(centers_list,x,y)
        new_centers_list = update_centers(clusters_list,centers_list)
    
    score = cluster_score(clusters_list,new_centers_list)
    new_total_score = sum(score)
#    print(new_total_score)
    if total_score < 0:
        total_score = new_total_score
        lowest_centers_list = new_centers_list
        lowest_clusters_list = clusters_list
        continue
    if new_total_score < total_score:
        total_score = new_total_score
        lowest_centers_list = new_centers_list
        lowest_clusters_list = clusters_list
    if loops == 100:
        break

print(total_score)

x_center = []
y_center = []

for center in lowest_centers_list:
    x_center.append(center[0])
    y_center.append(center[1])

plt.figure(1)
plt.scatter(x,y)
plt.scatter(x_center,y_center)

cluster_points = cluster_points_list(lowest_clusters_list,lowest_centers_list)
cluster_1 = np.array(cluster_points[0])
cluster_2 = np.array(cluster_points[1])
cluster_3 = np.array(cluster_points[2])
cluster_4 = np.array(cluster_points[3])
cluster_5 = np.array(cluster_points[4])

#print(a)
#print(new_centers_list)
#print(random_list)

plt.figure(2)
plt.scatter(cluster_1[:,0],cluster_1[:,1],c='r')
plt.scatter(cluster_2[:,0],cluster_2[:,1],c='c')
plt.scatter(cluster_3[:,0],cluster_3[:,1],c='g')
plt.scatter(cluster_4[:,0],cluster_4[:,1],c='y')
plt.scatter(cluster_5[:,0],cluster_5[:,1],c='b')
plt.scatter(x_center,y_center,c='k')
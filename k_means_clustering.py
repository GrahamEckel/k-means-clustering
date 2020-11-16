import random
import numpy as np
import pandas as pd

#the mnist dataset is stored in the keras library, so we pull it from there
from keras.datasets import mnist

#assigning arrays
(TrainX, TrainY), (TestX, TestY) = mnist.load_data()

# Representing each 28*28 greyscale image as a vector with size 784
TrainX = TrainX.reshape(len(TrainX),-1)
TestX = TestX.reshape(len(TestX), -1)

# Determine the number of clusters and iterations
K = len(np.unique(TestY))
iterations = 10

def kmeanscluster(data, K, P):
    # Initializing an iterator, a distance (from the point to the centroid) array, and our cost arrays
    rows = data.shape[0]
    distance = np.random.rand(rows)
    costs = np.random.rand(K)
    
    # Creating K random centroids from pulled data and setting a random seed for reproducibility
    random.seed(2020-11-14) #keep on when testing results
    rand = np.array([random.randint(0, 10000) for k in range(0, K)])
    centroids = data[rand,:]
    
    CostResult = []
    CentroidResult = []
    DataResult = []
    
    for i in range(0, P):
        for i in range(0, rows):
            #finding the nearest centre of every row in TrainX and assigning it to the nearest centroid
            distance[i] = np.argmin([np.sum((centroids[k,]-data[i,])**2) for k in range(len(centroids))]) 
    
        #taking the mean of all vectors associated with each cluster, updating new centroids to this vector 
        clustered = pd.DataFrame(data)   
        clustered['centroid number'] = distance
        means = clustered.groupby(['centroid number']).mean()
        centroids = means.to_numpy()
 
        DataResult.append(clustered)
        #DataResult = np.array(ClusteredResult)    
    
        CentroidResult.append(centroids)
        #CentroidResult = np.array(CentroidResult)   
 
        #compute the sum of the squared distances from each data point to its centroid (cost)
        for k in range(len(centroids)):  
            cluster = np.array(clustered.loc[clustered['centroid number'] == k])
            bycluster = cluster[:,:-1].copy()
            costs[k] = np.sum((bycluster-centroids[k,])**2)
        CostResult.append(sum(costs))
        #CostResult = np.array(CentroidResult)
     
    return DataResult, CentroidResult, CostResult

data, centroids, Jclust = kmeanscluster(TestX, K, iterations)

IterList = range(0,K)
for i, j in zip(Jclust, IterList):
    print('Iteration {} - Cost: {}'.format(j,i))

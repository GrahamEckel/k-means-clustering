import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#the mnist dataset is stored in the keras library, so we pull it from there
from keras.datasets import mnist

#assigning arrays
(TrainX, TrainY), (TestX, TestY) = mnist.load_data()

# Representing each 28*28 greyscale image as a vector with size 784
TrainX = TrainX.reshape(len(TrainX),-1)
TestX = TestX.reshape(len(TestX), -1)

def kmeanscluster(data, K):
    # Initializing an iterator, a distance (from the point to the centroid) array, and our cost arrays
    rows = data.shape[0]
    distance = np.random.rand(rows)
    costs = np.random.rand(K)
    
    # Creating K random centroids from pulled data and setting a random seed for reproducibility
    #random.seed(2020-11-14) #keep on when testing results
    rand = np.array([random.randint(0, rows) for k in range(0, K)])
    centroids = data[rand,:]
    
    CostResult = []
    CentroidResult = []
    DataResult = []
    
    for i in range(0, 6):
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
            costs[k] = (1/len(range(rows)))*np.sum((bycluster-centroids[k,])**2)
        CostResult.append(sum(costs))
        #CostResult = np.array(CentroidResult)
     
    return DataResult, CentroidResult, CostResult

start = time.time()
# Determine the number of clusters and iterations
K = 10
iterations = 20
data = TrainX

DataResult = []
CentroidResult = []
CostResult = []

IterList = range(1,iterations+1)
for i in (IterList):
    D, C, J = kmeanscluster(data, K)
    DataResult.append(D)
    CentroidResult.append(C)
    CostResult.append(J) 
end = time.time()
ElapsedTime = end-start

#at n-60 000, K=10 and iterations = 20, it takes about 12 minutes to train

#Finding the minimum and maximum cost vectors of our internal iterations
iter = len(CostResult)
arrCosts = np.random.rand(iter)
arrCostResult = np.array(CostResult)
for i in range(0,len(arrCostResult)):
    arrCosts[i] = min((arrCostResult[i,]))

low = min(arrCosts)
high = max(arrCosts)
arrlow = np.where(arrCosts == low)
arrhigh = np.where(arrCosts == high)

MinimumCost = arrCostResult[arrlow[0][0],]
MaximumCost = arrCostResult[arrhigh[0][0],]

print(MinimumCost)
print(MaximumCost)

#Min JClust, plotting previously found cost vectors
k = range(1,len(MinimumCost)+1)
plt.figure(figsize=(20,10))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(k,MinimumCost,'go--', linewidth=1.5, markersize=4)
plt.xlabel('Value of k',fontsize = 25)
plt.ylabel('Cost',fontsize = 25)
plt.title('Cost of Min Jclust per K Iterations', fontsize = 30)
plt.show

#Max JClust, plotting previously found cost vectors
k = range(1,len(MaximumCost)+1)
plt.figure(figsize=(20,10))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(k,MaximumCost,'go--', linewidth=1.5, markersize=4)
plt.xlabel('Value of k',fontsize = 25)
plt.ylabel('Cost',fontsize = 25)
plt.title('Cost of Max Jclust per K Iterations', fontsize = 30)

#Finding the calculated centroids at our min and max cost vectors
CentroidsMin = (CentroidResult[arrlow[0][0]])
CentroidsMax = (CentroidResult[arrhigh[0][0]])

#turning the 1*728 vector back into a 28*28 image and printing it
def show_digit(x):
    plt.axis('off')
    plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
    plt.show()
    return plt.show()

# Printing our centroids of min
LastCentInteration = CentroidsMin[len(CentroidsMin)-1]
for i in range(0,len(LastCentInteration)):
    c = LastCentInteration[i][np.newaxis]
    digit = show_digit(c)
    print(digit)

# Printing our centroids of max
LastCentInteration = CentroidsMax[len(CentroidsMax)-1]
for i in range(0,len(LastCentInteration)):
    c = LastCentInteration[i][np.newaxis]
    digit = show_digit(c)
    print(digit)
import random
import numpy as np

#the mnist dataset is stored in the keras library, so we pull it from there
from keras.datasets import mnist

(TrainX, TrainY), (TestX, TestY) = mnist.load_data()

#representing each 28*28 greyscale image as a vector with size 784
TrainX = TrainX.reshape(len(TrainX),-1)
TestX = TestX.reshape(len(TestX), -1)

#Determine the number of clusters
K = len(np.unique(TestY))
iterations = 1

def kmeanscluster(data, K):
    #creating K random centroids using dimensions and elements of our test data
    Centers = np.array([]).reshape(data.shape[1],0)
    for i in range(K):
        Centers = np.c_[Centers,data[random.randint(0,data.shape[0]-1)]]
    
    #the magic
    for i in range(iterations):
        euclid=np.array([]).reshape(data.shape[0],0)
        for k in range(K):
            dist=np.sum((data-Centers[:,k])**2,axis=1)
            euclid=np.c_[euclid,dist]
        c=np.argmin(euclid,axis=1)+1
        Cent={}
        for k in range(K):
            Cent[k+1]=np.array([]).reshape(data.shape[1],0)
        for k in range(data.shape[0]):
            Cent[c[k]]=np.c_[Cent[c[k]],data[k]]
        for k in range(K):
            Cent[k+1]=Cent[k+1].T
        for k in range(K):
            Centers[:,k]=np.mean(Cent[k+1],axis=0)
        final=Cent    
    return final
      
FinalCenters = kmeanscluster(TestX, K)
print(FinalCenters)
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:03:11 2017

@author: Rudra Balankari
"""
import numpy as np # importing numpy,pandas,sklearn,scipy,pyplot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#reading the data from the csv file using pandas 
dataIn = pd.read_csv('C:\Crime_Analysis.csv') 
x1=dataIn['Y'] #considering latitudes as x1
y1=dataIn['X'] #considering longitudes as y1

data = np.array(list(zip(x1, y1))).reshape(len(x1), 2)
 
#K Means Algorithm
distortions = []
#clusters in steps of 10
clusters = np.array([10,20,30,40,50,60,70,80,90,100]) 

for k in clusters:
    kmean = KMeans(n_clusters=k).fit(data)
    kmean.fit(data)
    distortions.append(sum(np.min(cdist(data, kmean.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
 
# Plotting elbow graph 
    
plt.plot(clusters, distortions, 'bx-')
# setting the label for x-axis
plt.xlabel('clusters')
# setting the label for x-axis
plt.ylabel('distortion')
# setting the title
plt.title('plot to select the number of clusters')
plt.show()

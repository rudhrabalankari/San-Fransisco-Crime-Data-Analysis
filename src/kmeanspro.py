# -*- coding: utf-8 -*-
"""
Created on  Fri Dec 01 10:41:20 2017

@author: rudhra balankari
"""

#K means clustering.
# Input  are latitude and longitude
# Usage: spark-submit kmeanspro.py <inputdatafile>

import sys
import numpy as np
from pyspark import SparkContext
from math import sqrt
from numpy import array

#  latitude and longitude is taken as input, calculate its distance from each centroid


def kmeanspro(line):
  latitudenlongitude = np.array(line).astype('float')
  #minimumdistance=infinity
  minimum = float("inf") 
  j = 1
  for i in centroidlist:
    distance = sqrt(sum((latitudenlongitude - i) ** 2))#calculating the distance between latitudenlongitude distances and assigning cluster value based on minimum distance	
    if distance < minimum:
	  minimum = distance
	  cluster = j
    j = j + 1
  return (cluster, latitudenlongitude)
 
# data is written to CSV file.
def toCSVLine(data):
    
  return ','.join(str(d) for d in data) 
#execution starts here
if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: spark-submit Kmeanspro.py <inputdatafile>"
    exit(-1)

  sc = SparkContext(appName="K means Clustering")

  input = sc.textFile(sys.argv[1]) #input file is read
#datapreprocessing
  header = input.first()  # indicates the names of the column 
  
  inputWithoutusingHeader = input.filter(lambda line : line!= header) #filtering the data without header using the filter
  
  
  rddRawdata = inputWithoutusingHeader.map(lambda line: line.split(',')) #spliting the given line at delimiter ',' on input without header
  
  rddfilterdata =  rddRawdata.filter(lambda x: x[7] is not u'') #delete the row for which latitude is empty.
  
  noOfRows = rddfilterdata.count() # counting number of rows in rdd
  
  
  ############# KMeamns implementation starts here.#############
  k = 40 #found using elbow method using a saperate program.
  noOfIterations = 30
  fraction = ((k+30)/(noOfRows*0.1)) * 0.1 #  Extra rows 
  
  sample = rddfilterdata.sample(True, fraction, 2300)
  
  
  list = sample.map(lambda line: (line[7], line[6])).take(k)
  
  # Now we convert the lists as float.
  centroidlist = array(list).astype('float')
  
  for i in range(noOfIterations): 
    rddCluster = rddfilterdata.map(lambda line: kmeanspro([line[7], line[6]])) #latitudes and longitudes
	# Calculate new set of centroids by grouping the points belonging to same cluster
	# and taking their average
    rddAggregate = rddCluster.aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1), lambda a,b: (a[0] + b[0], a[1] + b[1]))# Calculating new set of centroids by grouping the points
    rddAverage = rddAggregate.mapValues(lambda v: v[0]/v[1])        
    centroidlist = array(rddAverage.values().collect()).astype('float')  # Re-initialize centroid list with average of cluster

  rddFinal = rddfilterdata.map(lambda line: kmeanspro([line[7], line[6]]))    # Final classification 
  
  rddFinal.collect()
    
  lines = rddFinal.map(toCSVLine)
  
  lines.saveAsTextFile('SanfransiscoKmeans') #output file saved here
  sc.stop()

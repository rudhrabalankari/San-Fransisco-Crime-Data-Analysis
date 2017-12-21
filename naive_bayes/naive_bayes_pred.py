# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 14:52:51 2017

@author: rudhra balankari
"""
# input is category

# usage: spark-submit Naive_Bayes_Pred.py <category>
#
import sys
import numpy as np
from pyspark import SparkContext

########main execution starts here#############
if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: Naive_Bayes_Pred.py <category>"
    exit(-1)

  sc = SparkContext(appName="Naive_Bayes_Pred")
  
  # arg 1 is taken and read as input into the test category
  testCategory = sys.argv[1]
  
  # we have four parts of csv files generated and taken as input
  input = sc.textFile("Crime_Analysis_part_0.csv,Crime_Analysis_part_1.csv,Crime_Analysis_part_2.csv,Crime_Analysis_part_3.csv")
  # first line contains the headings
  
  header = input.first()
  # input without using header
  inputWithoutusingHeader = input.filter(lambda line : line!= header)
  
  inputRdd = inputWithoutusingHeader.map(lambda line: line.split(',')) #input rdd
    
  noOfRows = inputRdd.count() #counting no of rows in input rdd
  
  cluster_Occurances = inputRdd.map(lambda line: (line[8], float(1))).reduceByKey(lambda a,b: np.add(a,b).astype('float'))
  
  cluster_Occurances_Count = cluster_Occurances.mapValues(lambda a: (a/noOfRows))#clusteroccurancescount
  
  Norm_ClusterCountMap = cluster_Occurances_Count.collectAsMap()
  
  clusterCountMap=cluster_Occurances.collectAsMap()
  
  Hour_Occurances = inputRdd.map(lambda line: (line[4].split(':')[0], float(1))).reduceByKey(lambda a,b: np.add(a,b).astype('float')) #hour occurances
  
  hour_Occurances_Count = Hour_Occurances.mapValues(lambda a: (a/noOfRows)) #Houroccurancescount
  
  norm_Hour_Map = hour_Occurances_Count.collectAsMap() #norm Hour
  
  clusterHourMap=Hour_Occurances.collectAsMap() #cluster hour map
  
  distinctCategories = inputRdd.map(lambda line: line[1]).distinct().count()
  
  filteredInput = inputRdd.filter(lambda line : line[1]== testCategory)
  
  categoryClusterCount = filteredInput.map(lambda line: (line[8], float(1))).reduceByKey(lambda a,b: np.add(a,b).astype('float')).collectAsMap()
  categoryHourCount = filteredInput.map(lambda line: (line[4].split(':')[0], float(1))).reduceByKey(lambda a,b: np.add(a,b).astype('float')).collectAsMap()
  
  def findClusters(cluster, value):   #function for finding clusters of categories
    alpha = 1
    probabcluster = Norm_ClusterCountMap[cluster]
    try:
        clusterCategoryCount = categoryClusterCount[cluster]
    except KeyError:
        clusterCategoryCount = 0
        
    probabCategory_cluster = float(clusterCategoryCount+alpha)/float(value+ distinctCategories)
    posteriorProbab = probabCategory_cluster * probabcluster
    return (posteriorProbab, cluster)
	
  def findTime(hour, value):    #function for finding time
    alpha = 1
    probabHour = norm_Hour_Map[hour]
    try:
        hourCatCnt = categoryHourCount[hour]
    except KeyError:
        hourCatCnt = 0
    probabCategory_hour = float(hourCatCnt+alpha)/float(value+ distinctCategories)
    posteriorProbab = probabCategory_hour * probabHour
    return (posteriorProbab, hour)
  
  probClusters = cluster_Occurances.map(lambda line: findClusters(line[0], line[1]))
  probHours = Hour_Occurances.map(lambda line: findTime(line[0], line[1]))
  clusterList = probClusters.takeOrdered(10, key = lambda x: -x[0])
  hourList = probHours.takeOrdered(10, key = lambda x: -x[0])
  
  print "Top Areas: "  #printing Top Areas
  
  for coeff in clusterList:
    print coeff
	
  print "Top Hours: "   #Printing Top Hours
  
  for coeff in hourList:
    print coeff
      
  sc.stop()

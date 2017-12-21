# SF Crime Classification using Implemented Naive Bayes Algorithm
# Naive_Bayes.py
#
# Standalone Python/Spark program to predict and classify of crime categories given month, hour, weekday and cluster#
# Prints accuracy and outputs clasification results as actual label, predicted label
# Usage: spark-submit Naive_Bayes.py <inputFile>
# splits the input file as train and test splits by classifying odd weekdays as train data and even weekdays as test data
# Example usage: spark-submit Naive_Bayes.py Crime_Analysis_part_0.csv,Crime_Analysis_part_1.csv,Crime_Analysis_part_2.csv,Crime_Analysis_part_3.csv
#
import sys
import numpy as np
from pyspark import SparkContext
from datetime import *



if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: Naive_Bayes.py <datafiles>"
    exit(-1)

  sc = SparkContext(appName="NaiveBayes")
  input = sc.textFile(sys.argv[1])
  # seperate header from the file
  header = input.first()
  # Filter the input on which processing has to be performed
  inputWithoutHeader = input.filter(lambda line : line!= header)
  # Split the values in each record
  totalRdd = inputWithoutHeader.map(lambda line: line.split(','))
  
  # Custom function used for train test split
  def customMapper(line):
    if((date(int(line[3].split('/')[2]),int(line[3].split('/')[0]),int(line[3].split('/')[1])).isocalendar()[1])%2 == 0):
        return (0,line)
    else:
        return (1,line)

  mappedRdd = totalRdd.map(lambda line: customMapper(line)).cache()
  # Assign odd weeks to train split and even weeks to test split
  trainRdd = mappedRdd.filter(lambda x: x[0]==1).map(lambda x: x[1])
  testRdd = mappedRdd.filter(lambda x: x[0]==0).map(lambda x: x[1])
  
  # Steps added to increase train data as compared to test data
  splits = testRdd.randomSplit([0.4, 0.6])
  inputRdd = trainRdd.union(splits[0])
  testRdd = splits[1]
  
  noOfRows = inputRdd.count()
  categoryCount = inputRdd.map(lambda line: (line[1], float(1))).reduceByKey(lambda a,b: np.add(a,b).astype('float'))
  normalizedCategoryCount = categoryCount.mapValues(lambda a: (a/noOfRows))
  # Store posterior probabilities for each category
  normCategoryCountMap = normalizedCategoryCount.collectAsMap()
  
  # Calculate number of distinct weekdays in the input file
  distinctWeekdays = inputRdd.map(lambda line: line[2]).distinct().count()
  # Calculate number of distinct month in the input file
  distinctMonths = inputRdd.map(lambda line: line[3].split('/')[0]).distinct().count()
  # Calculate number of distinct hours in the input file
  distinctHours = inputRdd.map(lambda line: line[4].split(':')[0]).distinct().count()
  # Calculate number of distinct clusters in the input file
  distinctClusters = inputRdd.map(lambda line: line[8]).distinct().count()
  
  # Count occurrences of all available Category, Weekday counts. Used in calculating probability of a particular category given weekday.
  catWeekdayCount = inputRdd.map(lambda line: ((line[1],line[2]), float(1))).reduceByKey(lambda a,b: np.add(a,b).astype('float'))
  catWeekdayMap=catWeekdayCount.collectAsMap()
  # Count occurrences of all available Category, Month counts.
  catMonthCount = inputRdd.map(lambda line: ((line[1],line[3].split('/')[0]), float(1))).reduceByKey(lambda a,b: np.add(a,b).astype('float'))
  catMonthMap=catMonthCount.collectAsMap()
  # Count occurrences of all available Category, Hour counts.
  catHourCount = inputRdd.map(lambda line: ((line[1],line[4].split(':')[0]), float(1))).reduceByKey(lambda a,b: np.add(a,b).astype('float'))
  catHourMap=catHourCount.collectAsMap()
  # Count occurrences of all available Category, Cluster counts.
  catClusterCount = inputRdd.map(lambda line: ((line[1],line[8]), float(1))).reduceByKey(lambda a,b: np.add(a,b).astype('float'))
  catClusterMap=catClusterCount.collectAsMap()
    
  # Store count of occurrences of each category
  categoryCountMap=categoryCount.collectAsMap()
  
  # function to classify a record with the probability of occurrence of a crime category given month, hour, weekday, area. Returns predicted label.
  def classify(month, hour, weekday, cluster):
    alpha = 1
    maxProb = 0
    category_label = ''
    for key, value in categoryCountMap.iteritems():
        try:
            monthCount = catMonthMap[(key,str(month))]
        except KeyError:
            monthCount = 0
        try:
            hourCount = catHourMap[(key,str(hour))]
        except KeyError:
            hourCount = 0
        try:
            weekdayCount = catWeekdayMap[(key,weekday)]
        except KeyError:
            weekdayCount = 0
        try:
            clusterCount = catClusterMap[(key,str(cluster))]
        except KeyError:
            clusterCount = 0
        probabCat = normCategoryCountMap[key]
        probabHour_Cat = float(hourCount+alpha)/float(value+ distinctHours)
        probabWeek_Cat = float(weekdayCount+alpha)/float(value+ distinctWeekdays)
        probabMonth_Cat = float(monthCount+alpha)/float(value+ distinctMonths)
        probabCluster_Cat = float(clusterCount+alpha)/float(value+ distinctClusters)
        posteriorProbab = probabHour_Cat * probabWeek_Cat * probabMonth_Cat * probabCluster_Cat * probabCat
        if(posteriorProbab > maxProb):
            maxProb = posteriorProbab
            category_label = key
	return(category_label)
  
  # Each record in test data is classified with a predicted label
  label_pred = testRdd.map(lambda line: classify(line[3].split('/')[0], line[4].split(':')[0], line[2], line[8]))
  # Get actual labels for the test records
  label_actual = testRdd.map(lambda line: line[1])
  
  # Initialize counter to note instances labels being corrected accurately
  correct_labels = sc.accumulator(0)
  for label_a, label_p in zip(label_actual.collect(), label_pred.collect()):
    if(label_a == label_p):
        correct_labels.add(1)
		
  print "Accuracy is: "+ str((float(correct_labels.value)/float(testRdd.count()))*100)
  
  # Merge actual labels and predicted labels into single rdd to write to output file
  labelsAndPredictions = testRdd.map(lambda line: line[1]).zip(label_pred)
  
  # Custom function to write to output files
  def toCSVLine(data):
	return ','.join(str(d) for d in data)
	
  lines = labelsAndPredictions.map(toCSVLine)
  # Output is written to NBClassification folder
  lines.saveAsTextFile('NBClassification')
      
  sc.stop()

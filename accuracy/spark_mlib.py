# SF Crime Classification using Spark MLlib Naive Bayes Algorithm
# Spark_MLlib.py
#
# Standalone Python/Spark program to predict and classify of crime categories given month, hour, weekday and cluster#
# Prints accuracy of clasification results
# Usage: spark-submit Spark_MLlib.py <inputFiles>
# splits the input file as train and test splits by classifying odd weekdays as train data and even weekdays as test data
# Example usage: spark-submit Spark_MLlib.py Crime_Analysis_part_0.csv,Crime_Analysis_part_1.csv,Crime_Analysis_part_2.csv,Crime_Analysis_part_3.csv
#
import sys
import numpy as np
from pyspark import SparkContext
from datetime import *
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "spark-submit Spark_MLlib.py <datafiles>"
    exit(-1)

  sc = SparkContext(appName="MLlib Naive Bayes")
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
  inputRdd = mappedRdd.filter(lambda x: x[0]==1).map(lambda x: x[1])
  testRdd = mappedRdd.filter(lambda x: x[0]==0).map(lambda x: x[1])
  
  # Collect distinct crime categories present in the input
  crimeCategories = totalRdd.map(lambda line: line[1]).distinct().collect()
  crimeCategories.sort()
  
  # For each category assign a numerical label for NB algorithm to perform classification
  categories = {}
  for category in crimeCategories:
    categories[category] = float(len(categories))
  
  # Hashing used to convert categorical input features to numeric values
  htf = HashingTF(5000)
  
  # Perform feature extraction on train and test data splits to feed the data to algorithm
  trainingData = inputRdd.map(lambda x: LabeledPoint(categories[x[1]],htf.transform([x[2], x[3].split('/')[0], x[4].split(':')[0], x[8]])))
  testingSet = testRdd.map(lambda x: htf.transform([x[2], x[3].split('/')[0], x[4].split(':')[0], x[8]]))
  
  # Train the model on the train split. Classifies a record with the probability of occurrence of a crime category given month, hour, weekday, area.
  model = NaiveBayes.train(trainingData, 1.0)
  # Use the trained model to predict test data. Returns predicted labels for each record.
  predictions = model.predict(testingSet)
  
  # Get actual labels for the test records
  label_actual = testRdd.map(lambda x: categories[x[1]])
  
  # Initialize counter to note instances labels being corrected accurately
  correct_labels = sc.accumulator(0)
  for label_a, label_p in zip(label_actual.collect(), predictions.collect()):
    if(label_a == label_p):
        correct_labels.add(1)
  
  print "Accuracy is: "+ str((float(correct_labels.value)/float(predictions.count()))*100)
  
  sc.stop()

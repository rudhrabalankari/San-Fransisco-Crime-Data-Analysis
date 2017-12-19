# San-Fransisco-Crime-Data-Analysis
San Francisco city is most known for being a technology hub and a Tourist destination. However, the crime in the city is also increasing at an alarming rate. The aim of our project is to predict top locations and time using category of crime,classify the category of crime that occurred, given the information of time and location . These results would be helpful for SF police to understand the underlying patterns of the crimes  and aids in public safety by alerting the police to target specific locations at specific times. It helps in proper utilization of resources and in alleviating crimes.
Link for the Project : https://sites.google.com/uncc.edu/sf-crime-classification/home?authuser=2
K means Clustering
K-means clustering is implemented using pyspark
source code : Kmeanspro.py 
The latitudes and longitudes are taken as input from the csv file and k=40 no of clusters is determined using elbow method and the output is obtained in the form of cluster,latitude,longitude.
input files and Kmeans.py file are moved into  dsba hadoop cluster using winscp
input files are moved in HDFS using the following commands:
hadoop fs -put Police_Department_Incidents.csv /user/lchinnam/kmeans/input/ (directory is made)
Kmeanspro.py is run in spark for the input files using the command:
spark-submit Kmeanspro.py /user/lchinnam/kmeans/input/*
The output is written to folder named Sanfransiscokmeans in the hdfs 
elbow method is run in python(syder) for the elbow plot
##Naive Bayes for getting top areas and time using categories as input
Naive Bayes is executed using pyspark
K means is written to multiple files which is taken as input 
Merge K means output with Chicago_Crimes_updated.csv. 
The input to Naive Bayes are in parts named Crime_Analysis_part_0.csv,Crime_Analysis_part_1.csv,Crime_Analysis_part_2.csv,Crime_Analysis_part_3.csv
input files and Naive_Bayes_Pred.py are moved into  dsba hadoop cluster using winscp
Naive_Bayes_Pred.py is executed in spark with input category  
spark-submit Naive_Bayes_Pred.py category 
input paths are already given in the python code.
The Output is obtained with top 10 areas of clusters  and time.
##Execution of the program for Classification of crime categories
Merge the Kmeans output with input file to include cluster numbers in the input file for Naive Bayes
Copy the input files to HDFS in dsba clusters
hadoop fs -put Crime_Analysis_part_0.csv /user/lchinnam/Naivebayes/input/
hadoop fs -put /home/cloudera/Crime_Analysis_part_1.csv /user/lchinnam/Naivebayes/input/
hadoop fs -put /home/cloudera/Crime_Analysis_part_2.csv /user/lchinnam/Naivebayes/input/
hadoop fs -put /home/cloudera/Crime_Analysis_part_3.csv /user/lchinnam/Naivebayes/input/
Execute the implemented Naive Bayes program by running below command
spark-submit Naive_Bayes.py --inputFiles
(Ex: spark-submit Naive_Bayes.py Crime_Analysis_part_0.csv,Crime_Analysis_part_1.csv,Crime_Analysis_part_2.csv,Crime_Analysis_part_3.csv)
The output of actual and predicted labels written to NBClassification folder. Accuracy of prediction is printed on the console.  
Execute the MLlib Naive Bayes program by running below command
spark-submit Spark_MLlib.py --inputFiles
(Ex: spark-submit Spark_MLlib.py Crime_Analysis_part_0.csv,Crime_Analysis_part_1.csv,Crime_Analysis_part_2.csv,Crime_Analysis_part_3.csv)
The accuracy of predicting correct labels on the test split is printed on the console.

#!/usr/bin/python
#Project 2 Part 3 (Logistic Regression)

#Load necessary packages (Spark related libraries only - no sklearn)
from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import (count, col)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (OneHotEncoder, StringIndexer, VectorAssembler)
from pyspark.ml import Pipeline
import sys

if __name__ == "__main__":
	#Start the Spark session
	spark = SparkSession\
		.builder\
		.master("local[*]")\
		.appName("Predict Adult Salary")\
		.getOrCreate()
	
	#Read and load census data (training and testing)
	df_train = spark.read.load('../input/adult_train.csv', 
		format='com.databricks.spark.csv', 
        header='false', 
        inferSchema='true').cache()
	
	df_test = spark.read.load('../input/adult_test.csv', 
		format='com.databricks.spark.csv', 
        header='false', 
        inferSchema='true').cache()

	#Rename and clean up columns
	education_num = col("`education.num`")
	capital_gain = col("`capital.gain`")
	capital_loss = col("`capital.loss`")
	hours_per_week = col("`hours.per.week`")
	marital_status = col("`marital.status`")
	native_country = col("`native.country`")

	df_train = df.withColumn("education_num", education_num).drop(education_num)\
   		.withColumn("capital_gain", capital_gain).drop(capital_gain)\
   		.withColumn("capital_loss", capital_loss).drop(capital_loss)\
    	.withColumn("hours_per_week", hours_per_week).drop(hours_per_week)\
    	.withColumn("marital_status", marital_status).drop(marital_status)\
    	.withColumn("native_country", native_country).drop(native_country)
    
	#Sort columns into numerical and categorical types
	num_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
	cat_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
	all_cols = num_cols + cat_cols
	
	#Handle categorical columns
	stages = []
	for col in cat_cols:
    	#Use StringIndexer for indexing of the columns
    	stringIndexer = StringIndexer(
       		inputCol = col, 
        	outputCol = col + "_indx")
    	#OneHotEncoder to convert categorical columns
    	encoder = OneHotEncoder(
       		inputCols = [stringIndexer.getOutputCol()], 
        	outputCols = [col + "_vec"])
    	stages += [stringIndexer, encoder]
    
	#Convert label into label indices using StringIndexer
	label_index = StringIndexer(
    	inputCol = "income", 
   		outputCol = "label")
	stages += [label_index]

	assembler_inputs = [c + "_vec" for c in cat_cols] + num_cols
	assembler = VectorAssembler(
    	inputCols = assembler_inputs, 
    	outputCol = "features")
	stages += [assembler]

	#Create the pipeline
	pipeline1 = Pipeline(stages=stages)

	#Run the transformations
	df_train = pipeline1.fit(df_train).transform(df_train)

	#Combine back columns
	final_cols = ["label", "features"] + all_cols
	df_train = df_train.select(final_cols)

	#Set up logistic regression model
	logReg = LogisticRegression(
    	maxIter = 10,
    	regParam = 0.05,
    	labelCol = "label",
    	featuresCol = "features")
	
	#Fit model
	logModel = logReg.fit(df_train)
	
	#Evaluate model
	pred = logModel.transform(df_test)
    evaluator = BinaryClassificationEvaluator(labelCol = "label")
    print(evaluator.evaluate(pred))
	
	spark.stop()
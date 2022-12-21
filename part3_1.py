#!/usr/bin/python
#Project 2 Part 3 (Logistic Regression)

#Load necessary packages (Spark related libraries only - no sklearn)
from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import sys

if __name__ == "__main__":
	#Start the Spark session
	spark = SparkSession\
		.builder\
    		.master("local[*]")\
    		.appName('census')\
    		.getOrCreate()

	#Establish columns names and data types when reading in data
	schema1 = StructType([
    		StructField("age", IntegerType(), True),
    		StructField("workclass", StringType(), True),
    		StructField("fnlwgt", IntegerType(), True),
    		StructField("education", StringType(), True),
    		StructField("education_num", IntegerType(), True),
    		StructField("marital_status", StringType(), True),
    		StructField("occupation", StringType(), True),
    		StructField("relationship", StringType(), True),
    		StructField("race", StringType(), True),
    		StructField("sex", StringType(), True),
    		StructField("capital_gain", IntegerType(), True),
    		StructField("capital_loss", IntegerType(), True),
    		StructField("hours_per_week", IntegerType(), True),
    		StructField("native_country", StringType(), True),
    		StructField("salary", StringType(), True)])
	
	#Read and load census data (both training and testing sets)
	df_train = spark.read.csv('../project2/data/adult_train.csv', header='false', schema = schema1)
	df_test = spark.read.csv('../project2/data/adult_test.csv', header='false', schema = schema1)
    
   	#Sort columns into numerical and categorical types (excluding "salary" column)
	num_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
	cat_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
	
	indexers = [StringIndexer(inputCol = column, outputCol = column + "-index") for column in cat_cols]
	encoder = OneHotEncoder(inputCols = [indexer.getOutputCol() for indexer in indexers], outputCols = ["{0}-encoded".format(indexer.getOutputCol()) for indexer in indexers])
	
	assembler = VectorAssembler(
    		inputCols = encoder.getOutputCols(),
    		outputCol = "categorical-columns")
	
	pipeline = Pipeline(stages = indexers + [encoder, assembler])
	df_train = pipeline.fit(df_train).transform(df_train)
	df_test = pipeline.fit(df_test).transform(df_test)
	
	assembler = VectorAssembler(inputCols = ["categorical-columns", *num_cols], outputCol = "features")
	df_train = assembler.transform(df_train)
	df_test = assembler.transform(df_test)
	
	indexer = StringIndexer(inputCol = "salary", outputCol = "label")
	df_train = indexer.fit(df_train).transform(df_train)
	df_test = indexer.fit(df_test).transform(df_test)
	df_train.limit(10).toPandas()["label"]
	
	lr = LogisticRegression(featuresCol = "features", labelCol = "label")
	model = lr.fit(df_train)
	pred = model.transform(df_test)
	evaluator = BinaryClassificationEvaluator(labelCol = "label")
	print(evaluator.evaluate(pred))
	spark.stop()

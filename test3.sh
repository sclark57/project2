#!/bin/bash
source ../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /project2/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /project2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../project2/data/adult_test.csv /sort/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../project2/data/adult_train.csv /sort/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./part3_1.py hdfs://$SPARK_MASTER:9000/sort/input/
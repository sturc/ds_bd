# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# WordCount for K8s


# %%
from pyspark.sql import SparkSession


# %%
inputFile = "hdfs:///data/ghEmployees.txt"


# %%
outputFile = "hdfs:///tmp/jwcsturm2.txt"


# %%
#create a SparkSession without local master and app name
spark = (SparkSession.builder.getOrCreate())
# read file 
spark.sparkContext.setLogLevel("ERROR")
input = spark.sparkContext.textFile(inputFile)
counts = input.flatMap(lambda line : line.split(" ")).map(lambda word : [word, 1]).reduceByKey(lambda a, b : a + b)


# %%
# write the result to hdfs
counts.saveAsTextFile(outputFile)


# %%
print(counts.collect())


# %%
spark.stop()


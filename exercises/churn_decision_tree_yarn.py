#!/usr/bin/env python
# coding: utf-8

# # The Decision Tree on the Churn Dataset in Spark

# In[4]:


from pyspark.sql import DataFrameReader
from pyspark.sql import SparkSession
from pyspark.ml.feature import IndexToString, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier


# ## Select the churn file 

# In[5]:


inputFile = "hdfs:///data/churn.csv"


# ## Create the Spark Session 

# In[6]:


#create a SparkSession
spark = (SparkSession
       .builder
       .master("yarn") 
       .appName("ChurnDecisionTree")
       .getOrCreate())
# create a DataFrame using an ifered Schema 
df = spark.read.option("header", "true")        .option("inferSchema", "true")        .option("delimiter", ";")        .csv(inputFile)   


# ## Data Preparation
# ### Transform labels into index

# In[7]:


df.printSchema()

labelIndexer = StringIndexer().setInputCol("LEAVE").setOutputCol("label").fit(df)
collegeIndexer = StringIndexer().setInputCol("COLLEGE").setOutputCol("COLLEGE_NUM").fit(df)
# TODO add additional indexer for string attributes


#  ### Build the feature vector

# In[8]:


featureCols = df.columns.copy()
featureCols.remove("LEAVE")
featureCols.remove("COLLEGE")
featureCols.remove("REPORTED_SATISFACTION")
featureCols.remove("REPORTED_USAGE_LEVEL")
featureCols.remove("CONSIDERING_CHANGE_OF_PLAN")
featureCols = featureCols +["COLLEGE_NUM"]
print(featureCols)
# TODO add additinal columns to feature vector


# ### Build the feature Vector Assembler

# In[9]:


assembler =  VectorAssembler(outputCol="features", inputCols=list(featureCols))


# ### Convert indexed labels back to original labels

# In[10]:


predConverter = IndexToString(inputCol="prediction",outputCol="predictedLabel",labels=labelIndexer.labels)


# ## Do the Data Preparation

# In[11]:


labeledData = labelIndexer.transform(df)
# TODO add the other additional indexer
indexedLabedData = collegeIndexer.transform(labeledData)
labeledPointData = assembler.transform(indexedLabedData)


# ### Spliting the dataset into train and test set

# In[12]:


splits = labeledPointData.randomSplit([0.6, 0.4 ], 1234)
train = splits[0]
test = splits[1]


# ## Build the decision tree model

# In[ ]:


# TODO Optimize the properties 
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", impurity="entropy")
dtModel = dt.fit(train)


# ## Build an evaluator

# In[ ]:


evaluator =  BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction", metricName="areaUnderROC")


# ## Do the prediction 

# In[ ]:


predictions = dtModel.transform(test)
predictionsConverted = predConverter.transform(predictions)


# ## Evaluate / Test the Model 

# In[ ]:


predictionsConverted.select("prediction", "label", "predictedLabel", "LEAVE", "features").show()
# Select (prediction, true label) and compute test error.
   
accuracy = evaluator.evaluate(predictions)
print("Test Error = " ,(1.0 - accuracy))

spark.stop()



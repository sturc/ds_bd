from pyspark.sql.types import BooleanType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyparsing import col
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.evaluation import MulticlassMetrics
from helpers.helper_functions import translate_to_file_string

inputFile = translate_to_file_string("data/Boston_Housing_Data.csv")

# main program
if __name__ == "__main__":
   #create a SparkSession
   spark = (SparkSession
       .builder
       .appName("ChurnDataPreprocessing")
       .getOrCreate())
   # create a DataFrame using an ifered Schema 
   df = spark.read.option("header", "true") \
       .option("inferSchema", "true") \
       .option("delimiter", ";") \
       .csv(inputFile) \
       .withColumn("CATBOOL", expr("CAT").cast(BooleanType()))
   print(df.printSchema())

   featureCols = df.columns.copy()
   featureCols.remove("MEDV")
   featureCols.remove("CAT")
   featureCols.remove("CATBOOL") 
   # print(featureCols)
   assembler =  VectorAssembler(outputCol="features", inputCols=featureCols)
   
   # Prepare training and test data.
   splits = df.randomSplit([0.9, 0.1 ], 12345)
   training = splits[0]
   test = splits[1]
        
   #Support Vector Machine Classifier
   lsvc = LinearSVC(labelCol="CAT",aggregationDepth=2, featuresCol="features",maxIter=1000
                    ,regParam=0.001, standardization=True ) 
  
   
   # Build the pipeline
   pipeline = Pipeline(stages= [assembler, lsvc] )

   #fit (train) the model
   pipeModel = pipeline.fit(training)
		
   #test the model
   predictions = pipeModel.transform(test)
   predictions.show()
   evaluator = BinaryClassificationEvaluator(labelCol="CAT",rawPredictionCol="rawPrediction", metricName="areaUnderROC")
   accuracy = evaluator.evaluate(predictions)
   print("Test Error",(1.0 - accuracy))

		
   #logistic regression
   lr = LogisticRegression(labelCol="CAT",featuresCol="features", maxIter=100, \
                           regParam=0.001, standardization=True, aggregationDepth=2)
   pipeline = Pipeline(stages=[assembler, lr] )

   # fit (train) the model and test the model
   predictions = pipeModel.transform(test)
   predictions.show()

   accuracy = evaluator.evaluate(predictions)
   print("Test Error LR = " , (1.0 - accuracy))
   

   # Logistic regression LBFGS

   featureSet = assembler.transform(df)
   featureSetLP = (featureSet.select(featureSet.CAT, featureSet.features)
      .rdd
      .map(lambda row: LabeledPoint(row.CAT, DenseVector(row.features))))

   splitsLP = featureSetLP.randomSplit([ 0.9, 0.1 ], 12345)
   trainingLP = splitsLP[0]
   testLP = splitsLP[1]
   # create alternative logistic Regression classifier with the right parameters
   
   # Build the model
   modelLRLB = LogisticRegressionWithLBFGS.train(trainingLP, numClasses=2)

   #help(featureSetLP)
   # Do the classification
   predictionAndLabels = testLP.map(lambda x : [float(modelLRLB.predict(x.features )), float(x.label) ])
  
   # evaluate the result
   metrics =  MulticlassMetrics(predictionAndLabels)
   print("Test Error LRLBFG =" , (1.0 - metrics.accuracy))
		
   spark.stop()
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from helpers.helper_functions import translate_to_file_string

inputFile = translate_to_file_string("data/Boston_Housing_Data.csv")

# main program
if __name__ == "__main__":
   #create a SparkSession
   spark = (SparkSession
       .builder
       .appName("BostonHousingRegression")
       .getOrCreate())
   # create a DataFrame using an ifered Schema 
   df = spark.read.option("header", "true") \
       .option("inferSchema", "true") \
       .option("delimiter", ";") \
       .csv(inputFile)
   print(df.printSchema())

   featureCols = df.columns.copy()
   featureCols.remove("MEDV")
   featureCols.remove("CAT")
   
   assembler =  VectorAssembler(outputCol="features", inputCols=featureCols)
   

		
   # Prepare training and test data.
   splits = df.randomSplit([0.9, 0.1 ], 12345)
   training = splits[0]
   test = splits[1]

   lr = LinearRegression(maxIter=1000, regParam= 0.001, elasticNetParam=0.8, featuresCol="features", labelCol="MEDV")
   pipeline = Pipeline(stages = [assembler, lr ])

   lrModel = pipeline.fit(training)
		
   # fit (train) the model and test the model
   predictionsLR = lrModel.transform(test)
   predictionsLR.show()

   evaluator = RegressionEvaluator(labelCol="MEDV",predictionCol="prediction", metricName="rmse")

   print("root mean square error = " , evaluator.evaluate(predictionsLR))
		
   spark.stop()
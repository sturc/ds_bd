"""Helper functions to pyspark data analysis"""

from pyspark.sql import Window
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col, monotonically_increasing_id, row_number


def print_df(sprkDF,num_rows=0): 
  """Pretty print spark dataframes in jupyter"""
  if num_rows > 0:
    new_df = sprkDF.limit(num_rows).toPandas() 
  else:
    new_df = sprkDF.toPandas()
  from IPython.display import display, HTML
  return HTML(new_df.to_html())  


def add_weight_col(dataframe, label_col='label', weight_col_name='classWeightCol'):  
  """Re-balancing (weighting) of records to be used in the logistic loss objective function"""
  num_negatives = dataframe.filter(col(label_col) == 0).count()
  dataset_size = dataframe.count()
  balancing_ratio = (dataset_size - num_negatives)/ dataset_size
  def calculate_weights (d):
    if (d == 0):
      return 1 * balancing_ratio
    else:
      return (1 * (1.0 - balancing_ratio))

  calculate_weights_udf = udf(lambda z: calculate_weights (z),DoubleType())

  weighted_dataframe = dataframe.withColumn(weight_col_name, calculate_weights_udf(col(label_col)))
  return weighted_dataframe

def print_confusion_matrix(spark, conf_matrix):
  """pretty printing the confusion matrix in the common format Row=Prediction Column=Label"""
  #print (list(map(list, zip(*conf_matrix.toArray().tolist()))))
  col_name_count = 0
  df_conf = spark.createDataFrame(list(map(list, zip(*conf_matrix.toArray().tolist()))))
  for col_name in df_conf.columns.copy():
    df_conf = df_conf.withColumnRenamed(col_name,"Label_"+str(col_name_count))
    col_name_count +=1
  new_col_names = df_conf.columns.copy()
  df_conf = df_conf.withColumn("monotonically_increasing_id", monotonically_increasing_id())
  window = Window.orderBy(col('monotonically_increasing_id'))
  df_conf = df_conf.withColumn('Prediction', row_number().over(window)-1)
  new_col_names.insert(0,"Prediction")
  df_conf = df_conf.select(new_col_names)
  return print_df(df_conf)




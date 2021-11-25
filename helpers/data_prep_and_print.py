"""Helper functions to pyspark data analysis"""

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col


def print_df(sprkDF): 
    """Pretty print spark dataframes in jupyter"""
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






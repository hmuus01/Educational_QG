# This command just considered all the 90 Million rows to one row
import os
import threading
import time
import pandas as pd
import pyspark
import pyspark.pandas as ps
# Import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
# Create SparkSession
from tqdm import tqdm
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
MAX_SEQ_LEN = 512
spark = SparkSession.builder \
      .master("local[1]") \
      .config('spark.executor.memory', '10g') \
      .config("spark.driver.memory", "30g") \
      .config('spark.driver.maxResultSize', '0') \
      .config('spark.memory.offHeap.size', '20g') \
      .appName("SparkByExamples.com") \
      .getOrCreate()

datapath_file = os.path.join('data','SCI', 'pdf', 'pdf_parses_36.jsonl')
datapath = os.path.join('data','SCI', 'pdf')

data_df_1 = spark.read.json(datapath_file)

schema = data_df_1.schema
data_df = spark.read.json(datapath, schema=schema)


print("Read file into json")

# df.printSchema()
# df.head()
data_df.show()
df1 = data_df[['abstract']]

abstract_df = df1[['abstract']]
# body_df = df1['body_text']
# sc = spark.sparkContext

start = time.time()

interval = 1000


def get_texts(df, label):
    counter = 0
    abstracts = []
    for row in tqdm(df.rdd.collect()):
        if len(row) > 0 :
            # print("checking abstract")
            # print(row)
            abstract = row[0]
            if not len(abstract):
                continue
            # print(abstract)
            # print(abstract[0])
            # print("----------------")
            text = abstract[0]['text'][:MAX_SEQ_LEN]
        else:
            continue
        abstracts.append(text)

        if counter%interval ==0 and counter//interval >=1:
            end = time.time()
            print(f'{label} - {counter//interval}: time: {end - start}')
            print('----------------')

        counter+=1
    return abstracts
abstracts = get_texts(abstract_df, 'abstracts')
# abstracts+= get_texts(body_df, 'body')
dfout = spark.createDataFrame(abstracts, "string").toDF("text")
print("finished loop")
dfout.show()

dfout.toPandas().to_csv('data/science_full/cs.csv', index=False)
print("done data")
#dfout.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("mydata.csv")



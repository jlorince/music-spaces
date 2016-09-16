"""
set HADOOP_HOME=U:\Users\jjl2228\hadoop
set PYSPARK_DRIVER_PYTHON="ipython"
"""

from pyspark.sql.types import *
import time
import pandas as pd

start = time.time()


artist_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/artist_data',header=None,names=['artist_id2','artist_name','total_scrobbles','unique_listeners'])
include = artist_data[(artist_data['total_scrobbles']>=10000)&(artist_data['unique_listeners']>=100)][['artist_id2']]
artist_ids = spark.createDataFrame(include)

#schema = StructType([StructField("user_id", IntegerType(), True),StructField("item_id", IntegerType(), True),StructField("artist_id", IntegerType(), True),StructField("scrobble_time", DateType(), True)])
#schema = StructType([StructField("user_id", IntegerType(), True),StructField("item_id", IntegerType(), True),StructField("artist_id", IntegerType(), True),StructField("scrobble_time", TimeStampType(), True)])
#df = spark.read.csv('P:/Projects/BigMusic/jared.rawdata/lastfm_scrobbles.txt',sep='\t',header=None,schema=schema)
df = spark.read.parquet('/P:/Projects/BigMusic/jared.parquet/scrobbles-date.parquet').select(["artist_id","scrobble_time"])

# schema_aid = StructType([StructField("artist_id", IntegerType(), True),StructField("artist_name", StringType(), True)])
# artist_ids = spark.read.csv('P:/Projects/BigMusic/jared.rawdata/artist_ids',sep='\t',header=None,schema=schema_aid)
# artist_ids = artist_ids.select("artist_id")

joined = df.join(artist_ids,df.artist_id == artist_ids.artist_id2).persist()

result = joined.groupBy(['artist_id','scrobble_time']).count()


result.show()
total_time = time.time() - start

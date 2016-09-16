"""
set HADOOP_HOME=U:\Users\jjl2228\hadoop
set PYSPARK_DRIVER_PYTHON="ipython"
"""

from pyspark.sql.types import *
import time

start = time.time()
#schema = StructType([StructField("user_id", IntegerType(), True),StructField("item_id", IntegerType(), True),StructField("artist_id", IntegerType(), True),StructField("scrobble_time", DateType(), True)])
schema = StructType([StructField("user_id", IntegerType(), True),StructField("item_id", IntegerType(), True),StructField("artist_id", IntegerType(), True),StructField("scrobble_time", TimestampType(), True)])


df = spark.read.csv('P:/Projects/BigMusic/jared.rawdata/lastfm_scrobbles.txt',sep='\t',header=None,schema=schema)


gn = spark.read.csv('gracenote_song_data',sep='\t',header=True,inferSchema=True).select(['songID','genre1','genre2','genre3'])

joined = df.join(gn,df.item_id == gn.songID)

result = joined.groupBy(['scrobble_time','genre1','genre2','genre3']).count()


result.show()
total_time = time.time() - start



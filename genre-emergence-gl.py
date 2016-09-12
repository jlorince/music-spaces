import graphlab as gl
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.parser import parse
from graphlab import aggregate as agg

rootdir = 'genres/'

### CONFIGURATION
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 12)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 12)
#gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
#gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)

nrows=None

scrobbles = gl.SFrame.read_csv('lastfm_scrobbles.txt',header=None,delimiter='\t',nrows=nrows)
scrobbles.rename({'X1':'uid','X2':'songID','X3':'aid','X4':'ts'})

gn = gl.SFrame.read_csv('gracenote_song_data',delimiter='\t',usecols=['songID','genre1','genre2','genre3'],nrows=nrows)
gn.save('gracenote_sframe')

# g1 = gn['genre1'].unique()
# g2 = gn['genre2'].unique()
# g3 = gn['genre3'].unique()

genres = {'genre1':gn['genre1'].unique(),'genre2':gn['genre2'].unique(),'genre3':gn['genre3'].unique()}

joined = scrobbles.join(gn,on='songID')

joined['ts'] = joined['ts'].apply(lambda x: parse(x))

ts = gl.TimeSeries(joined,index='ts')
ts.save('ts')

total_listens = ts.resample(dt.timedelta(days=1),agg.COUNT())
total_listens.save(rootdir+'_total_listens')

for level in ('genre1','genre2','genre3'):
    n = len(genres[level])
    for i,genre in enumerate(genres[level]):

        current = ts[ts[level]==genre].resample(dt.timedelta(days=1),agg.COUNT())
        #current.save(rootdir+level+'_'+genre)
        current.to_sframe().to_dataframe().to_pickle(rootdir+level+'_'+genre.replace('/','-')+'.pkl')
        print "{} - {}  ({}/{})".format(level,genre,i+1,n)

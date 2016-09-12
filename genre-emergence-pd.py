import pandas as pd
import numpy as np
import os
import glob
import time
import sys
import cPickle
import logging
import datetime
import os

# start = time.time()
# wall_time = 4 * 60 * 60 # 4hr
# time_buffer = 30 * 60 # 30min

# done = set()
# if len(sys.argv)>1:
#     logfiles = sys.argv[1:]
#     for fi in logfiles:
#         with open(fi) as f:
#             for line in f:
#                 line = line.strip().split('\t')
#                 done.add(line[-1].split()[0])

idx = int(sys.argv[1])
blocksize= 170

inputdir = '/N/dc2/scratch/jlorince/scrobbles-complete/'
outputdir = '/N/dc2/scratch/jlorince/genre_stuff/'
gn_path = '/N/dc2/scratch/jlorince/gracenote_song_data'
#gn_path = '/Users/jaredlorince/Dropbox/Research/misc.data/gracenote_song_data'


if os.path.exists(outputdir+'genre_data_'+str(idx)):
    sys.exit()


now = datetime.datetime.now()
log_filename = now.strftime('genres_%Y%m%d_%H%M%S.log')
logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
rootLogger = logging.getLogger()
# fileHandler = logging.FileHandler(log_filename)
# fileHandler.setFormatter(logFormatter)
# rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)


offset = idx*blocksize
files = sorted(glob.glob(inputdir+'*'))[offset:offset+blocksize]
if len(files)==0:
    sys.exit()
n_users = len(files)


gn = pd.read_table(gn_path,usecols=['songID','genre1','genre2','genre3']).dropna().set_index('songID')

daterange = pd.date_range(start='2005-07-01',end='2012-12-31',freq='D')

#genres = {'genre1':sorted(gn['genre1'].unique()),'genre2':sorted(gn['genre2'].unique()),'genre3':sorted(gn['genre3'].unique())}
#cPickle.dump(genres,open(outputdir+'gn_genres.pkl','w'))
genres = cPickle.load(open(outputdir+'gn_genres.pkl'))

result = pd.DataFrame(0.,index=daterange,columns=genres['genre1']+genres['genre2']+genres['genre3'])
# if len(done)==0:
#     result = pd.DataFrame(0.,index=daterange,columns=genres['genre1']+genres['genre2']+genres['genre3'])
# else:
#     result = pd.read_pickle(outputdir+'genre_data')

for i,f in enumerate(files):
    user_start = time.time()
    if f in done:
        continue
    df = pd.read_table(f,sep='\t',header=None,names=['item_id','artist_id','scrobble_time'],parse_dates=['scrobble_time']).join(gn,on='item_id',how='left')
    for level in genres:
        vars()['df_'+level] = df.set_index('scrobble_time').groupby([pd.TimeGrouper(freq='D'),level]).count()['item_id'].unstack().reindex(daterange,columns=genres[level])
    concat = pd.concat([df_genre1,df_genre2,df_genre3],axis=1).fillna(0)

    result += concat

    rootLogger.info("{} ({}/{}, {:.1f})".format(f,i+1,n_users,time.time()-user_start))
    #time_elapsed = time.time() - start
    # if time_elapsed >= (wall_time-(time_buffer)):
    #     result.to_pickle(outputdir+'genre_data')
    #     sys.exit()

result.to_pickle(outputdir+'genre_data_'+str(idx))

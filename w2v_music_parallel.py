import gzip,os,glob,cPickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm as tq

import time,datetime
class timed(object):
    def __init__(self,desc='command',pad='',**kwargs):
        self.desc = desc
        self.kwargs = kwargs
        self.pad = pad
    def __enter__(self):
        self.start = time.time()
        print '{} started...'.format(self.desc)
    def __exit__(self, type, value, traceback):
        if len(self.kwargs)==0:
            print '{}{} complete in {}{}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),self.pad)
        else:
            print '{}{} complete in {} ({}){}'.format(self.pad,self.desc,str(datetime.timedelta(seconds=time.time()-self.start)),','.join(['{}={}'.format(*kw) for kw in self.kwargs.iteritems()]),self.pad)


dim = 200
win = 5    
min_count = 100

scrobble_path = 'P:/Projects/BigMusic/jared.IU/scrobbles-complete/'
#base_output_path = 'P:/Projects/BigMusic/jared.data/d2v/blocks/'
base_output_path = 'P:/Projects/BigMusic/jared.data/d2v/songs/'

# if not os.path.exists(base_output_path+'docs.txt.gz'):
#     with gzip.open(base_output_path+'docs.txt.gz','w') as fout, gzip.open(base_output_path+'indices.txt.gz','w') as indices:
#         files = sorted(glob.glob(scrobble_path+'*.txt'))
#         for fi in tq(files):
#             doc = ' '.join([line.split('\t')[1] for line in open(fi)])
#             fout.write(doc+'\n')
#             userid = fi[fi.rfind('\\')+1:-4]
#             indices.write(userid+'\n')
# documents = [doc for doc in tq(TaggedLineDocument(base_output_path+'docs.txt.gz'))]

def process_artist_blocks(fi):
    artists = [line.split('\t')[1] for line in open(fi)]
    last = None
    blocks = []
    for a in artists:
        if a != last:
            blocks.append(a)
        last = a
    doc = ' '.join(blocks)
    userid = fi[fi.rfind('\\')+1:-4]
    return userid,doc

def get_songs(fi):
    songs = [line.split('\t')[0] for line in open(fi)]
    doc = ' '.join(songs)
    userid = fi[fi.rfind('\\')+1:-4]
    return userid,doc
    
if __name__=='__main__':

    output_path = base_output_path+'{}-{}-{}'.format(dim,win,min_count)
    if os.path.exists(output_path):
        raise Exception('path exists!')
    else:
        os.mkdir(output_path)


    import math
    from gensim.models.doc2vec import Doc2Vec,TaggedLineDocument


    # if not os.path.exists(base_output_path+'docs_artist_blocks.txt.gz'):
    #     procs = mp.cpu_count()
    #     pool = mp.Pool(procs)
    #     files = glob.glob(scrobble_path+'*.txt')
    #     n = len(files)
    #     chunksize = int(math.ceil(n / float(procs)))
    #     with gzip.open(base_output_path+'docs_artist_blocks.txt.gz','w') as fout, gzip.open(base_output_path+'indices.txt.gz','w') as indices:
    #         for userid,doc in tq(pool.imap_unordered(process_artist_blocks,files,chunksize=100),total=n):
    #             fout.write(doc+'\n')
    #             indices.write(userid+'\n')

    if not os.path.exists(base_output_path+'docs_songs.txt.gz'):
        procs = mp.cpu_count()
        pool = 30 #mp.Pool(procs)
        files = glob.glob(scrobble_path+'*.txt')
        n = len(files)
        chunksize = int(math.ceil(n / float(procs)))
        with gzip.open(base_output_path+'docs_songs.txt.gz','w') as fout, gzip.open(base_output_path+'indices.txt.gz','w') as indices:
            for userid,doc in tq(pool.imap_unordered(get_songs,files,chunksize=100),total=n):
                fout.write(doc+'\n')
                indices.write(userid+'\n')

    with timed('Loading docs'):
        #documents = [doc for doc in tq(TaggedLineDocument(base_output_path+'docs_artist_blocks.txt.gz'))]
        documents = [doc for doc in tq(TaggedLineDocument(base_output_path+'docs_songs.txt.gz'))]
    with timed('Running model'):
        model = Doc2Vec(documents, size=dim, window=win, min_count=min_count,workers=procs)

    with timed('Saving results'):
        # from sklearn.preprocessing import Normalizer
        # nrm = Normalizer('l2')
        # normed = nrm.fit_transform(model.docvecs.doctag_syn0)
        # words_normed = nrm.fit_transform(model.syn0)

        # np.save(output_path+'/doc_features_normed-{}-{}-{}.npy'.format(dim,win,min_count),normed)
        # np.save(output_path+'/word_features_normed-{}-{}-{}.npy'.format(dim,win,min_count),words_normed)
        model.save(output_path+'/model-{}-{}-{}'.format(dim,win,min_count))

    if False:

        with timed('Sanity checks...'):
            dpath = 'P:/Projects/BigMusic/jared.data/d2v/artist_dict.pkl'
            if not os.path.exists(dpath):
                artist_dict = {}
                for line in tq(open('P:/Projects/BigMusic/jared.rawdata/lastfm_itemlist.txt')):
                    line = line.split('\t')
                    if line[1]=='0':
                        artist_dict[line[2]] = line[0]
                cPickle.dump(artist_dict,open(dpath,'wb'))
            else:
                artist_dict = cPickle.load(open(dpath))
            id_dict = {v:k for k,v in artist_dict.iteritems()}

            def get_most_similar(artist_name):
                result = model.most_similar(artist_dict[artist_name])
                return [(id_dict[x[0]],x[1]) for x in result]

            for artist in ('radiohead','metallica','katy+perry','the+beatles','beethoven','modest+mouse','nas'):
                print artist,get_most_similar(artist)

    if True:

        with timed('Sanity checks...'):

            df = pd.read_pickle('P:/Projects/BigMusic/jared.rawdata/gracenote_song_data').set_index('songID')

            def get_most_similar(aid):
                result = model.most_similar(aid)
                sim = [x[0] for x in result]
                for a in sim:
                    try:
                        row = df.idx[aid]
                        print [row[c] for c in ['gn_artist','gn_song','genre1','genre2','genre3']]
                    except:
                        print '???'
                    
                
            for aid in df.head(1000).sample(25).index:
                row = df.ix[aid]
                print '------\n'+' - '.join([row[c] for c in ['gn_artist','gn_song','genre1','genre2','genre3']])+'\n------\n'
                get_most_similar(aid)

    try:
        pool.close()
    except:
        pass



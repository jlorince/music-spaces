from gensim.models.doc2vec import Doc2Vec,TaggedLineDocument
import gzip,os,glob,cPickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm as tq

dim = 200
win = 10    
min_count = 10
workers = mp.cpu_count()

scrobble_path = 'P:/Projects/BigMusic/jared.IU/scrobbles-complete/'
base_output_path = 'P:/Projects/BigMusic/jared.data/d2v/blocks/'

output_path = base_output_path+'{}-{}-{}'.format(dim,win,min_count)
if os.path.exists(output_path):
    raise Exception('path exists!')
else:
    os.mkdir(output_path)

# if not os.path.exists(base_output_path+'docs.txt.gz'):
#     with gzip.open(base_output_path+'docs.txt.gz','w') as fout, gzip.open(base_output_path+'indices.txt.gz','w') as indices:
#         files = sorted(glob.glob(scrobble_path+'*.txt'))
#         for fi in tq(files):
#             doc = ' '.join([line.split('\t')[1] for line in open(fi)])
#             fout.write(doc+'\n')
#             userid = fi[fi.rfind('\\')+1:-4]
#             indices.write(userid+'\n')
# documents = [doc for doc in tq(TaggedLineDocument(base_output_path+'docs.txt.gz'))]

if not os.path.exists(base_output_path+'docs_artist_blocks.txt.gz'):
    with gzip.open(base_output_path+'docs_artist_blocks.txt.gz','w') as fout, gzip.open(base_output_path+'indices.txt.gz','w') as indices:
        files = sorted(glob.glob(scrobble_path+'*.txt'))
        for fi in tq(files):
            artists = [line.split('\t')[1] for line in open(fi)]
            last = None
            blocks = []
            for a in tq(artists):
                if a != last:
                    blocks.append(a)
                last = a
            doc = ' '.join(blocks)
            fout.write(doc+'\n')
            userid = fi[fi.rfind('\\')+1:-4]
            indices.write(userid+'\n')
documents = [doc for doc in tq(TaggedLineDocument(base_output_path+'docs_artist_blocks.txt.gz'))]



%time model = Doc2Vec(documents, size=dim, window=win, min_count=min_count,workers=workers)

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



from sklearn.preprocessing import Normalizer
nrm = Normalizer('l2')
normed = nrm.fit_transform(model.docvecs.doctag_syn0)
words_normed = nrm.fit_transform(model.syn0)

np.save(output_path+'/doc_features_normed-{}-{}-{}.npy'.format(dim,win,min_count),normed)
np.save(output_path+'/word_features_normed-{}-{}-{}.npy'.format(dim,win,min_count),words_normed)
model.save(output_path+'/model-{}-{}-{}'.format(dim,win,min_count))

for artist in ('radiohead','metallica','britney+spears','weezer'):
    print get_most_similar(artist)




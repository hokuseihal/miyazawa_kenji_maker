'''save word2vecmodel'''

from gensim.test.utils import common_texts,get_tmpfile
from gensim.models import Word2Vec
from janome.tokenizer import Tokenizer 
import os
def makex2vmodel(textpath='ginga.txt'):
    if os.path.exists('kenjiw2v.model'):
        print("word2vec load kenjiw2v.model")
        return Word2Vec.load('kenjiw2v.model')
    else:
        # read txt
        with open(textpath) as f:
            lines=f.readlines()
        # wakati text
        t=Tokenizer()
        wakatilist=[t.tokenize(line,wakati=True) for line in lines]
        # make model
        w2vmodel=Word2Vec(wakatilist,size=100)
        w2vmodel.save('kenjiw2v.model')
        print('word2vec save kenjiw2v.model')

        index2word=w2vmodel.wv.index2word

        return w2vmodel
makex2vmodel()
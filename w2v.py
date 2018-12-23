'''save word2vecmodel'''

from gensim.test.utils import common_texts,get_tmpfile
from gensim.models import Word2Vec
from janome.tokenizer import Tokenizer 
import os
def makex2vmodel(flow):
    if os.path.exists('kenjiw2v.model'):
        print("word2vec load kenjiw2v.model")
        model=Word2Vec.load('kenjiw2v.model')
        return model
    else:
        # read txt
        print('word2vec making model.......')

        wakatilist=[ word[0] for word in flow]
        # make model
        w2vmodel=Word2Vec(wakatilist,min_count=1)
        w2vmodel.train(wakatilist,epochs=w2vmodel.iter,total_examples=w2vmodel.corpus_count,total_words=23612)
        w2vmodel.save('kenjiw2v.model')
        print('word2vec save kenjiw2v.model')

        index2word=w2vmodel.wv.index2word

        return w2vmodel
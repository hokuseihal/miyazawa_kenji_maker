'''
input : dictionary vocab and part
output : model fitted
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from gensim.models import word2vec_corpusfile
from gensim.models import  word2vec
import numpy as np
import io
class vocab:
    def __init__(self,dicti):
        self.max_length=40
        self.step=1
        self.weightpath='weight_vocabs.h5'
        self.w2vbinpath='w2v.bin'
        self.vocablist=[s[0] for s in dicti]
        self.vocabs=sorted(list(set(self.vocablist)))
        self.vocab_indices=dict((i,p) for p,i in enumerate(self.vocabs))
        self.indices_vocab=dict((p,i) for p,i in enumerate(self.vocabs))
        self.optimizer=RMSprop()
        self.model=Sequential()
        self.model.add(LSTM(128,input_shape=(self.max_length,len(self.vocabs))))

        self.model.add(Dense(len(self.vocabs),activation='softmax'))
        self.model.compile(self.optimizer,loss='categorical_crossentropy')

        if os.path.exists(self.w2vbinpath):
            self.w2v_model=KeyedVectors.load_word2vec_format(self.w2vbinpath,binary=True)
        else:


    def fit(self):
        pass
    def predict(self):
        pass
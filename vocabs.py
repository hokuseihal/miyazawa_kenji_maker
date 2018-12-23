'''
input : dictionary vocab and part
output : model fitted
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from gensim.models import  word2vec
import numpy as np
import io
import os
from  w2v import makex2vmodel

class vocab:
    def __init__(self, flow_ided):
        self.max_length=10
        self.weightpath='weight_vocabs.h5'
        self.wordn=max(flow_ided)+1
        self.step=3
        self.sentences=[]
        self.nextwords=[]
        for i in range(0,len(flow_ided)-self.max_length):
            self.sentences.append(flow_ided[i:i+self.max_length])
            self.nextwords.append(flow_ided[i+self.max_length])
        # build model
        self.model=Sequential()
        self.model.add(LSTM(128,input_shape=(self.max_length,self.wordn)))
        self.model.add(Dense(self.wordn,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer=RMSprop())
        if os.path.exists(self.weightpath):
            print('vocab:load wight')
            self.model.load_weights(self.weightpath)
        else:
            print('vocab:fitting.................')
            self.fitvocab(flow_ided)




    def fitvocab(self,flow_ided):
        x=np.zeros((len(self.sentences),self.max_length,self.wordn),dtype=np.bool)
        y=np.zeros((len(self.sentences),self.wordn),dtype=np.bool)
        for i ,sentence in enumerate(self.sentences):
            for t ,word in enumerate(sentence):
                x[i,t,word]=True
                y[i,self.nextwords[i]]=True
        self.model.fit(x,y,batch_size=63,epochs=1)
        self.model.save_weights(self.weightpath)
    def predict(self):
        # test
        x=np.random.randint(0,2,(len(self.sentences),self.max_length,self.wordn))
        print(x)
        return (self.model.predict(x))
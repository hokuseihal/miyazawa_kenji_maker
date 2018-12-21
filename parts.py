
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import io
import re
import os

class part:

    def __init__(self,dicti):
        self.max_length=120
        self.step=1
        self.weightpath= 'weight_parts.h5'
        self.part_list=[s[1] for s in dicti]
        self.parts=sorted(list(set(self.part_list)))
        self.part_indices=dict((i,p) for p,i in enumerate(self.parts))
        self.indices_part=dict((p,i) for p,i in enumerate(self.parts))
        # build model
        self.optimizer=RMSprop()
        self.model=Sequential()
        self.model.add(LSTM(128,input_shape=(self.max_length,len(self.parts))))
        self.model.add(Dense(128))
        self.model.add(Dense(len(self.parts),activation='softmax'))
        self.model.compile(self.optimizer,loss='categorical_crossentropy')
        if  os.path.exists(self.weightpath) :
            self.model.load_weights(self.weightpath)
        else:
            self.partfit()


    def partfit(self,dicti):

        sentenses=[]
        next_parts=[]
        #cut text
        for i in range(0,len(self.part_list)-self.max_length,self.step):
            sentenses.append(self.part_list[i:i+self.max_length])
            next_parts.append(self.part_list[i+self.max_length])
        # Vectorization
        x=np.zeros((len(sentenses),self.max_length,len(self.parts)),dtype=np.bool)
        y=np.zeros((len(sentenses),len(self.parts)),dtype=np.bool)

        for i ,sentense in enumerate(sentenses):
            for t,part in enumerate(sentense):
                x[i,t,self.part_indices[part]]=1
            y[i,self.part_indices[part]]=1
        self.model.fit(x,y,batch_size=63,epochs=3)
        self.model.save_weights(self.weightpath)

    def predict(self):
        #test
        x=np.random.randint(0,2,(1,self.max_length,len(self.parts)))
        #test end
        print(x)
        print(self.model.predict(x))
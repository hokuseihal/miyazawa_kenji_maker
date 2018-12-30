'''
input : dictionary vocab and part
output : model fitted
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import os
from showhis import showhis
from sklearn.model_selection import train_test_split


class vocab:
    def __init__(self, flow_ided, table, epoch=60,gen=True):
        self.batchsize=128
        self.epoch = epoch
        self.table = table
        self.max_length = 80
        self.flow_ided=flow_ided
        self.weightpath = 'weight_vocabs.h5'
        self.wordn = max(flow_ided) + 1
        self.step = 3
        # build model
        self.model = Sequential()
        self.model.add(LSTM(self.batchsize, input_shape=(self.max_length, self.wordn)))
        self.model.add(Dense(self.wordn, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
        if os.path.exists(self.weightpath):
            print('vocab:load wight')
            self.model.load_weights(self.weightpath)
        elif not gen:
            print('vocab:fitting.................')
            self.fitvocab()
        elif gen:
            print('vocab GEN fitting............')
            self.model.fit_generator(self.fitvocab_eng2(),steps_per_epoch=len(self.flow_ided)/self.batchsize,epochs=self.epoch)

    def fitvocab(self):
        self.sentences = []
        self.nextwords = []
        for i in range(0, len(self.flow_ided) - self.max_length):
            self.sentences.append(self.flow_ided[i:i + self.max_length])
            self.nextwords.append(self.flow_ided[i + self.max_length])
        x = np.zeros((len(self.sentences), self.max_length, self.wordn), dtype=np.bool)
        y = np.zeros((len(self.sentences), self.wordn), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, word in enumerate(sentence):
                x[i, t, word] = True
                y[i, self.nextwords[i]] = True
        (x_train, x_test, y_train, y_test) = train_test_split(x, y)
        his_fit = self.model.fit(x_train, y_train, batch_size=63, epochs=self.epoch, validation_data=(x_test, y_test))
        showhis(his_fit, title='vocabs:loss and validation_loss')
        self.model.save_weights(self.weightpath)


    def fitvocab_eng(self):
        for reti in range(0,len(self.flow_ided)-self.batchsize,self.batchsize):
            yield self.onehotter(self.flow_ided[reti:reti+self.max_length])[0],self.onehotter([self.flow_ided[reti+self.max_length]])[0]

    def fitvocab_eng2(self):
        for reti in range(0,len(self.flow_ided)-self.batchsize-1,self.batchsize):
            x = np.zeros((self.batchsize, self.max_length, self.wordn), dtype=np.bool)
            y = np.zeros((self.batchsize, self.wordn), dtype=np.bool)
            for i in range(self.batchsize):
                x[i,:,:]=self.onehotter(self.flow_ided[reti+i:reti+i+self.max_length])
                y[i,:]=self.onehotter([self.flow_ided[reti+self.max_length+i]])
            yield x,y
    def predict(self, sentenses):
        idlist = sentenses[-self.max_length:]
        return (self.model.predict(self.onehotter(idlist)))

    def onehotter(self, idlist):
        x = np.zeros((1, len(idlist), self.wordn), dtype=np.bool)
        for t, word in enumerate(idlist):
            x[0, t, word] = True
        return x

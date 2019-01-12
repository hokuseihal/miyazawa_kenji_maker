'''
input : dictionary vocab and part
output : model fitted
'''
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.optimizers import RMSprop
import numpy as np
import os
from showhis import showhis
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.utils import plot_model


class vocab:
    def __init__(self, flow_ided, table, epoch=60):
        self.epoch = epoch
        self.table = table

        self.max_length = 80
        self.epoch=epoch
        self.weightpath = 'weight_vocabs.h5'
        self.wordn = len(table) + 1
        self.step = 3
        self.sentences = []
        self.nextwords = []
        for i in range(0, len(flow_ided) - self.max_length):
            self.sentences.append(flow_ided[i:i + self.max_length])
            self.nextwords.append(flow_ided[i + self.max_length])
        # build model
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.max_length, self.wordn),kernel_regularizer=regularizers.l2(0.01),dropout=0.25,return_sequences=True))
        for i in range(2):
            self.model.add(LSTM(128, kernel_regularizer=regularizers.l2(0.01), dropout=0.25, return_sequences=True))
        self.model.add(LSTM(128, kernel_regularizer=regularizers.l2(0.01), dropout=0.25))
        self.model.add(Dense(self.wordn, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
        if os.path.exists(self.weightpath):
            print('vocab:load wight')
            self.model.load_weights(self.weightpath)
        else:
            print('vocab:fitting.................')
            self.fitvocab()
        plot_model(self.model, to_file='vocabmodel.png')

    def fitvocab(self):
        x = np.zeros((len(self.sentences), self.max_length, self.wordn), dtype=np.bool)
        y = np.zeros((len(self.sentences), self.wordn), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, word in enumerate(sentence):
                x[i, t, word] = True
                y[i, self.nextwords[i]] = True

        (x_train, x_test, y_train, y_test) = train_test_split(x, y)
        his_fit = self.model.fit(x_train, y_train, batch_size=63, epochs=self.epoch, validation_data=(x_test, y_test))
        showhis(his_fit, title='vocabs:loss_and_validation_loss')
        self.model.save_weights(self.weightpath)

    def predict(self, sentenses):
        idlist = sentenses[-self.max_length:]
        return (self.model.predict(self.onehotter(idlist)))

    def onehotter(self, idlist):
        x = np.zeros((1, self.max_length, self.wordn), dtype=np.bool)
        for t, word in enumerate(idlist):
            x[0, t, word] = True
        return x

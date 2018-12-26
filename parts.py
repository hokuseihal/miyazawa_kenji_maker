from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import io
import re
import os
import flow2table


class part:

    def __init__(self, flow):
        self.table=table = flow2table.flow2table(flow)
        self.max_length = 80
        self.step = 1
        self.weightpath = 'weight_parts.h5'
        self.part_list = [s[1] for s in flow]
        self.parts = sorted(list(set(self.part_list)))
        self.part_indices = dict((i, p) for p, i in enumerate(self.parts))
        self.indices_part = dict((p, i) for p, i in enumerate(self.parts))
        # build model
        self.optimizer = RMSprop()
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.max_length, len(self.parts))))
        self.model.add(Dense(128))
        self.model.add(Dense(len(self.parts), activation='softmax'))
        self.model.compile(self.optimizer, loss='categorical_crossentropy')
        if os.path.exists(self.weightpath):
            self.model.load_weights(self.weightpath)
            print('parts:load weight')
        else:
            print('parts:fitting...............')
            self.partfit()

    def partfit(self):

        sentenses = []
        next_parts = []
        # cut text
        for i in range(0, len(self.part_list) - self.max_length, self.step):
            sentenses.append(self.part_list[i:i + self.max_length])
            next_parts.append(self.part_list[i + self.max_length])
        # Vectorization
        x = np.zeros((len(sentenses), self.max_length, len(self.parts)), dtype=np.bool)
        y = np.zeros((len(sentenses), len(self.parts)), dtype=np.bool)

        for i, sentense in enumerate(sentenses):
            for t, _part in enumerate(sentense):
                x[i, t, self.part_indices[part]] = 1
            y[i, self.part_indices[_part]] = 1
        self.model.fit(x, y, batch_size=63, epochs=3)
        self.model.save_weights(self.weightpath)

    def predict(self, id_sentence):
        # input : id list
        # output : a part:string
        id_sentence = id_sentence[-self.max_length:]
        return self.indices_part[np.argmax(self.model.predict(self.onehotter(id_sentence)))]

    def onehotter(self, id_sentence):
        x = np.zeros((1, self.max_length, len(self.parts)), dtype=np.bool)
        for t, id in enumerate(id_sentence):
            x[0, t, self.part_indices[self.table[id][1]]] = True
        return x


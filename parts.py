from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
from sklearn.model_selection import train_test_split
import os
from showhis import showhis


class part:


    def __init__(self, table, flow, epoch=60):
        self.epoch = epoch
        self.flow = flow
        self.table = table
        self.max_length = 20

        self.step = 1
        self.weightpath = 'weight_parts.h5'
        self.parts = ['フィラー', '副詞', '助動詞', '助詞', '動詞', '名詞', '形容詞', '感動詞', '接続詞', '接頭詞', '記号', '連体詞']
        self.part_list = [s[1] for s in flow if s[1] in self.parts]
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
                x[i, t, self.part_indices[_part]] = 1
            y[i, self.part_indices[next_parts[i]]] = 1
        (x_train, x_test, y_train, y_test) = train_test_split(x, y)
        his_fit = self.model.fit(x_train, y_train, batch_size=63, epochs=self.epoch, validation_data=(x_test, y_test))
        showhis(his_fit, title='part:loss and validation_loss')

        self.model.save_weights(self.weightpath)
        self.accuracy_test()
    def predict(self, id_sentence, n):
        # input : id list
        # output : a part:string
        xsentence = id_sentence[-self.max_length:]
        return self.indices_part[np.argmax(self.model.predict(self.onehotter(xsentence)))]


    def onehotter(self, xinlist_onehot):
        x = np.zeros((1, self.max_length, len(self.parts)), dtype=np.bool)
        for t, id in enumerate(xinlist_onehot):
            x[0, t, self.part_indices[self.table[id][1]]] = True
        return x

    def accuracy_test(self):
        import  re
        print('accuracy_check')
        with open('ryunosuke.txt',encoding='shift-jis') as f:
            text = f.read()
        text=re.sub(r'《.*?》|［.*?］','',text)
        from janome.tokenizer import Tokenizer
        text = Tokenizer().tokenize(text)
        part_list = [(str(str(ppart).split()[1]).split(',')[0]) for ppart in text if
                     (str(str(ppart).split()[1]).split(',')[0]) in self.parts]
        t = 0
        l = 0
        sum = 0
        sentenses = []
        next_parts = []
        # cut text
        for i in range(0, len(part_list) - self.max_length, self.step):
            sentenses.append(part_list[i:i + self.max_length])
            next_parts.append(part_list[i + self.max_length])
        # Vectorization
        x = np.zeros((len(sentenses), self.max_length, len(self.parts)), dtype=np.bool)
        y = np.zeros((len(sentenses), len(self.parts)), dtype=np.bool)

        for i, sentense in enumerate(sentenses):
            for t, _part in enumerate(sentense):
                x[i, t, self.part_indices[_part]] = 1
            y[i, self.part_indices[next_parts[i]]] = 1
        xx = self.model.predict(x)
        for si in range(len(sentenses)):
            if np.argmax(y[si]) == np.argmax(xx[si]):
                t += 1
            else:
                l += 1
            sum += 1
        print()
        print('accuracy:', t / sum, l / sum)

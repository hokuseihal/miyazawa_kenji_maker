from janome.tokenizer import Tokenizer
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import io
import re
import os

def predict(sentense):
    # return next vocab list
    pass


def fitting(url='https://www.aozora.gr.jp/cards/000081/files/456_15050.html'):
    max_length=120
    step=1
    sentenses=[]
    next_parts=[]


    #get text
    textpath=get_file('ginga.txt',origin=url)
    with io.open(textpath, encoding='Shift_JIS') as f:
        text = f.read().lower()
    text = re.sub(r"<.*?>|（.*?）", "", text)
    parts=Tokenizer().tokenize(text)
    part_list=[(str(str(part).split()[1]).split(',')[0]) for part in parts]
    parts=sorted(list(set(part_list)))
    part_indices=dict((i,p) for p,i in enumerate(parts))
    indices_part=dict((p,i) for p,i in enumerate(parts))
    #cut text
    for i in range(0,len(part_list)-max_length,step):
        sentenses.append(part_list[i:i+max_length])
        next_parts.append(part_list[i+max_length])
    # Vectorization
    x=np.zeros((len(sentenses),max_length,len(parts)),dtype=np.bool)
    y=np.zeros((len(sentenses),len(parts)),dtype=np.bool)

    for i ,sentense in enumerate(sentenses):
        for t,part in enumerate(sentense):
            x[i,t,part_indices[part]]=1
        y[i,part_indices[part]]=1
    # build model
    model=Sequential()
    model.add(LSTM(128,input_shape=(max_length,len(parts))))
    model.add(Dense(len(parts),activation='softmax'))

    optimizer=RMSprop()
    model.compile(optimizer,loss='categorical_crossentropy')


    weightpass='weight_parts.h5'
    if  os.path.exists(weightpass):
        model.load_weights(weightpass)
    else:
        model.fit(x,y,batch_size=63,epochs=3)
        model.save_weights(weightpass)

fitting()
print('END')


import sys

from janome.tokenizer import Tokenizer
from keras.utils.data_utils import get_file
import numpy as np

import io
import re

import parts
import vocabs
import flow2table
import glob


def miyazawa():
    # making text and dictionary get text
    want_length = 100
    text=''
    # only test ---------------
    url = 'http://bible.salterrae.net/kougo/html/genesis.html'

    textpath = get_file('ginga.txt', origin=url)
    with io.open(textpath, encoding='Shift_JIS') as f:
       rawtext = f.read().lower()
    text += re.sub(r"<.*?>|（.*?）", "", rawtext)
    # only test end----------
    '''
    textlist=glob.glob('../html/*.html')
    for file in textlist:
        with io.open(file) as f:
            rawtext = f.read().lower()
        text += re.sub(r"<.*?>|（.*?）", "", rawtext)
    '''
    divided_text = Tokenizer().tokenize(text)
    flow = [((str(str(part).split()[0])), (str(str(part).split()[1]).split(',')[0])) for part in divided_text]
    table = flow2table.flow2table(flow)
    flow_ided = [table.index(drop) for drop in flow]

    # get models (fit)
    print('main:getting model')
    part_model = parts.part(flow)
    vocab_model = vocabs.vocab(flow_ided)
    print('main:predicting')
    # predict
    maxlength = vocab_model.max_length
    rand = np.random.randint(0, len(flow_ided) - maxlength)
    sentences = flow_ided[rand:rand + maxlength]
    for i in range(want_length):
        partpredict = part_model.predict(sentences)
        vocabpredict = vocab_model.predict(sentences)
        for j in range((vocabpredict.size)):
            vocabpredictid = (np.argsort(vocabpredict)[::-1])[0][j]
            tmp = table[vocabpredictid][1]
            tmp2=table[vocabpredictid]
            if table[vocabpredictid][1] == partpredict:
                sentences.append(vocabpredictid)
                # print console
                print(vocabpredictid)

                break


if __name__ == '__main__':
    miyazawa()

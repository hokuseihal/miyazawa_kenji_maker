from janome.tokenizer import Tokenizer
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer as toto

import io
import re

import parts
import vocabs
import flow2table


def miyazawa():
    # making text and dictionary get text

    # only test ---------------
    url = 'https://www.aozora.gr.jp/cards/000081/files/456_15050.html'

    textpath = get_file('ginga.txt', origin=url)
    with io.open(textpath, encoding='Shift_JIS') as f:
        text = f.read().lower()
    text = re.sub(r"<.*?>|（.*?）", "", text)
    # only test end----------

    divided_text = Tokenizer().tokenize(text)
    flow = [((str(str(part).split()[0])), (str(str(part).split()[1]).split(',')[0])) for part in divided_text]
    table = flow2table.flow2table(flow)
    flow_ided = [table.index(drop) for drop in flow]

    # get models (fit)
    print('main:getting model')
    part_model = parts.part(flow)
    vocab_model=vocabs.vocab(flow_ided)
    print('main:predicting')
    # predict
    for id in vocab_model.predict():
        print(table[id])
    # part_model.predict()
    # print here


if __name__ == "__main__":
    miyazawa()
    print('END')

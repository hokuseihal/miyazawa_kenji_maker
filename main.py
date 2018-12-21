from janome.tokenizer import Tokenizer
from keras.utils.data_utils import get_file
import io
import re

import parts
import vocabs
def miyazawa():
    # making text and dictionary get text


    #only test ---------------
    url='https://www.aozora.gr.jp/cards/000081/files/456_15050.html'

    textpath = get_file('ginga.txt', origin=url)
    with io.open(textpath, encoding='Shift_JIS') as f:
        text = f.read().lower()
    text = re.sub(r"<.*?>|（.*?）", "", text)
    # only test end----------


    divided_text = Tokenizer().tokenize(text)

    dicti = [((str(str(part).split()[0])),(str(str(part).split()[1]).split(',')[0])) for part in divided_text]
    # get models (fit)
    part_model=parts.part(dicti)
    partlist=part_model.parts
    vocab_model=vocabs.vocab(dicti)
    # predict
    part_model.predict()
    #print here

if __name__ == "__main__":
    miyazawa()
    print('END')

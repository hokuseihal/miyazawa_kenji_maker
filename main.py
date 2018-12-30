from janome.tokenizer import Tokenizer
import numpy as np
import os

import parts
import vocabs
import flow2table
import  re


def miyazawa(usepart=True):
    # making text and dictionary get text
    want_length = 100
    divided_text = []
    # only test --------------
    t = Tokenizer()
    folder = 'texts'
    files = os.listdir(folder)
    for file in files:
        with open(os.path.join(folder, file), encoding='shift_jis') as f:
            print('read: ', file)
            lines = f.readlines()
            for line in lines:
                if None!=re.match(r'\d{3}',line):
                    divided_text+=t.tokenize(line)[2:]
    # only test end----------
    list=['フィラー', '副詞', '助動詞', '助詞', '動詞', '名詞', '形容詞', '感動詞', '接続詞', '接頭詞', '記号', '連体詞']
    flow = [((str(str(part).split()[0])), (str(str(part).split()[1]).split(',')[0])) for part in divided_text if (str(str(part).split()[1]).split(',')[0]) in list]
    table = flow2table.flow2table(flow)
    flow_ided = [table.index(drop) for drop in flow]

    # get models (fit)
    print('main:getting model')
    part_model = parts.part(table, flow, epoch=2)
    vocab_model = vocabs.vocab(flow_ided, table, epoch=2)
    print('main:predicting')
    # predict
    maxlength = vocab_model.max_length
    rand = np.random.randint(0, len(flow_ided) - maxlength)
    sentences = flow_ided[rand:rand + maxlength]
    for i in range(want_length):
        partpredict = part_model.predict(sentences, rand + i)
        vocabpredict = vocab_model.predict(sentences)[0]
        for vocabidad in np.argsort(vocabpredict)[::-1]:
            if table[vocabidad][1] == partpredict:
                sentences.append(int(vocabidad))
                print(table[vocabidad][0], end="")
                break



if __name__ == '__main__':
    miyazawa()

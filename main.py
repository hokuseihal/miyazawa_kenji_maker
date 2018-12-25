from janome.tokenizer import Tokenizer
import numpy as np

import io
import re

import parts
import vocabs
import flow2table
import glob


def miyazawa():
    # making text and dictionary get text
    want_length = 10
    train_text=''
    run_text=''
    # only test ---------------
    with io.open('ginga.txt') as f:
       rawtext = f.read().lower()
    train_text += re.sub(r"<.*?>|（.*?）", "", rawtext)
    with io.open('yodaka.txt') as f:
        run_text=f.read()
    # only test end----------
    '''holly book
    textlist=glob.glob('../html/*.html')
    for file in textlist:
        with io.open(file) as f:
            rawtext = f.read().lower()
        text += re.sub(r"<.*?>|（.*?）", "", rawtext)
    '''
    train_flow = [((str(str(part).split()[0])), (str(str(part).split()[1]).split(',')[0])) for part in Tokenizer().tokenize(train_text)]
    run_flow=[((str(str(part).split()[0])), (str(str(part).split()[1]).split(',')[0])) for part in Tokenizer().tokenize(run_text)]
    table = flow2table.flow2table(train_flow+run_flow)
    train_flow_ided = [table.index(drop) for drop in train_flow]
    run_flow_ided=[table.index(drop) for drop in run_flow]
    # get models (fit)
    print('main:getting model')
    part_model = parts.part(train_flow,table)
    vocab_model = vocabs.vocab(train_flow_ided,table)
    print('main:predicting')
    # predict
    maxlength = vocab_model.max_length
    rand = np.random.randint(0, len(run_flow_ided) - maxlength)
    sentences = run_flow_ided[rand:rand + maxlength]
    for i in range(want_length):
        partpredict = part_model.predict(sentences)
        vocabpredict = vocab_model.predict(sentences)[0]
        for vocabidad in np.argsort(vocabpredict)[::-1]:
            if table[vocabidad][1]==partpredict:
                sentences.append(vocabidad)
                break


if __name__ == '__main__':
    miyazawa()

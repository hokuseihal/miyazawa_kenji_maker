from janome.tokenizer import Tokenizer
from keras.utils.data_utils import get_file
import io
import re
path = get_file(
    'ginga.txt',
    origin='https://www.aozora.gr.jp/cards/000081/files/456_15050.html')
with io.open(path, encoding='Shift_JIS') as f:
    text = f.read().lower()

text = re.sub(r"<.*?>|（.*?）", "", text)

t=Tokenizer()
s=text
w=t.tokenize(s)

for word in w:
    print(str(str(word).split()))
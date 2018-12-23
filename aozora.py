import re
from keras.utils.data_utils import get_file
import io
list=[]
with open('aozora.txt',encoding="utf-8_sig") as f:
    lines=f.readlines()
for line in lines:
    list.append(re.sub('".*','',line))
print(list)
url = 'https://www.aozora.gr.jp'+list[0]
textpath=get_file('t.txt',url)
with io.open(textpath, encoding='utf-8_sig') as f:
    text = f.read().lower()
text = re.sub(r"<.*?>|（.*?）", "", text)
print(text)
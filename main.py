import pickle

from lib.np import *
from cnn.model import  CNN
from cbow.util import word2vec

from keybert import KeyBERT
from typing import List
from konlpy.tag import Okt
from textrankr import TextRank

pkl_file = 'cbow/cbow_params-2.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

mi = -11.023438
mx = 10.265625
def regularization(x):
    return (x - mi) / (mx - mi + 1e-6)

cnn = CNN()
cnn.load_params('cnn/my_convnet_params-16.pkl')

def str2matrix(txt):
    input = txt.split(' ')
    ret = [[0.0]*128]*1024

    cnt = 0
    for i in range(len(input)):
        vec = word2vec(input[i], word_to_id, id_to_word, word_vecs)
        if vec is None:
            continue
        reg = regularization(vec)
        ret[i] = reg.tolist()
        cnt += 1
    print("correct : %d" % (cnt))

    return np.array([[ret]])

print("요약할 글의 정보를 입력해주세요.")
input = str(input())

mat = str2matrix(input)
ret = cnn.predict(mat)
ret = (-1 * ret[0]).argsort()


print("글의 주제와 관련 있는 5개의 키워드입니다.")
print("[My Model]")
for i in range(6):
    print(id_to_word[int(ret[i])], end=' ')
print()

print("[Key Bert]")
kb = KeyBERT()
ret_kb = kb.extract_keywords(input)
for i in ret_kb:
    print(i[0], end=' ')
print()

print("[Text Rankr]")
class OktTokenizer:
    okt: Okt = Okt()

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.okt.phrases(text)
        return tokens
okt = OktTokenizer()

tr = TextRank(okt)
print(tr.summarize(input, 1))
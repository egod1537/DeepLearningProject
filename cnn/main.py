import pickle

from lib.np import *
from cnn.model import  CNN
from cbow.util import word2vec

pkl_file = '../cbow/cbow_params-2.pkl'

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
cnn.load_params('my_convnet_params-16.pkl')

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

mat = str2matrix("사과 바나나 귤")
print(mat[0][0])

ret = cnn.predict(mat)
dict = ret[0]

dict = (-1 * dict).argsort()
print(dict)

for i in range(10):
    print(id_to_word[int(dict[i])], end=' ')

import time
import os
import json
import pickle

from lib.np import *
from lib.util import to_cpu
from cbow.util import word2vec

pkl_file = 'cbow_params-2.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

mi, mx = (1e9, -1e9)
vocab_size = len(word_to_id)

for word in word_to_id:
    for v in word2vec(word, word_to_id, id_to_word, word_vecs):
        mi = min(mi, v)
        mx = max(mx, v)

def regularization(x):
    return (x - mi) / (mx - mi + 1e-6)

print("mi : %.6f mx : %.6f" % (mi, mx))

DATA_PATH = "..\\..\\datas\\data"
SAVE_PATH = "..\\..\\datas\\temp"
MAX_COUNT = 16384

H = 1024
W = 128

sz = 0

start_time = time.time()
for (root, directories, files) in os.walk(DATA_PATH):
    for file in files:
        if sz % 100 == 0:
            elapsed_time = time.time() - start_time
            print("count : %d time : %.2f" % (sz, elapsed_time))
        sz += 1

        file_path = os.path.join(root, file)

        f = open(file_path, 'r', encoding='utf-8-sig')
        json_string = f.read()
        f.close()

        json_object = json.loads(json_string)

        input = str(json_object["input"]).split(' ')
        output = str(json_object["output"]).split(' ')

        ret = {"input": [[0.0] * W] * H, "output": [0.0] * vocab_size}

        for i in range(len(input)):
            vec = word2vec(input[i], word_to_id, id_to_word, word_vecs)
            reg = regularization(vec)
            ret["input"][i] = reg.tolist()

        for i in output:
            ret["output"][word_to_id[i]] += 1

        save_path = SAVE_PATH + "\\" + str(sz) + ".txt"
        with open(save_path, 'w', encoding='UTF-8-sig') as outfile:
            json.dump(ret, outfile, ensure_ascii=False)

        if sz >= MAX_COUNT:
            exit(0)
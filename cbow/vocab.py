import os
import json

PATH_VOCAB = "..\\..\\datas\\vocab"

def load_my_vocab():
    corpos = []
    word_to_id = {}
    id_to_word = {}

    cnt = 0
    id = 0
    for (root, directories, files) in os.walk(PATH_VOCAB):
        for file in files:
            file_path = os.path.join(root, file)

            f = open(file_path, 'r', encoding='utf-8-sig')
            json_string = f.read()
            f.close()

            words = json.loads(json_string)

            if cnt % 1000 == 0:
                print("load %d vocab-size : %d" % (cnt, len(word_to_id)))
            cnt += 1

            for word in words.split(' '):
                if not word in word_to_id:
                    word_to_id[word] = id
                    id_to_word[id] = word
                    id += 1
                corpos.append(word_to_id[word])

    return corpos, word_to_id, id_to_word


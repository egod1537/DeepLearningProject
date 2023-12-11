import os
import json
import random
import time

header = ["article", "book", "eassay"]

BASE_PATH = "..\..\datas"
DATA_PATH = "..\..\datas\data"
SAVE_PATH = "vocab"
MAX_COUNT = 100_000

sz = 0

pathes = []
for head in header:
    for PATH in ["training", "validation"]:
        path = BASE_PATH + "\\" + head + "\\" + "precompute\\" + PATH

        for (root, directories, files) in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                pathes.append(file_path)

random.shuffle(pathes)
pathes = pathes[:MAX_COUNT]
print("count %d" % (len(pathes)))

start_time = time.time()
for file_path in pathes:
    if sz % 1000 == 0:
        elapsed_time = time.time() - start_time
        print("count : %d time : %.2f" % (sz, elapsed_time))
    sz += 1

    f = open(file_path, 'r', encoding='utf-8-sig')
    json_string = f.read()
    f.close()

    json_object = json.loads(json_string)
    ret = json_object["input"] + " <no> " + json_object["output"]

    data_path = DATA_PATH + "\\" + str(sz) + ".txt"
    with open(data_path, "w", encoding='UTF-8-sig') as outfile:
        json.dump(json_object, outfile, ensure_ascii=False)

    save_path = BASE_PATH + "\\" + SAVE_PATH + "\\" + str(sz) + ".txt"
    with open(save_path, 'w', encoding='UTF-8-sig') as outfile:
        json.dump(ret, outfile, ensure_ascii=False)


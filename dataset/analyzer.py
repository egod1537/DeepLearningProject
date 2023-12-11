import os
import json
import time

header = ["article", "book", "eassay"]

BASE_PATH = "..\..\datas"

sz = 0

def get_token_size(str):
    return len(str.split(' '))

start_time = time.time()
ret = 0
for head in header:
    for PATH in ["training", "validation"]:
        path = BASE_PATH + "\\" + head + "\\" + "precompute\\" + PATH

        for (root, directories, files) in os.walk(path):
            for file in files:
                if sz % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print("count : %d max : %d time : %.2f" % (sz, ret, elapsed_time))
                sz += 1

                file_path = os.path.join(root, file)

                f = open(file_path, 'r', encoding='utf-8-sig')
                json_string = f.read()
                f.close()

                json_object = json.loads(json_string)
                ret = max(ret, get_token_size(json_object["input"]))
                ret = max(ret, get_token_size(json_object["output"]))
import os
import json

import numpy as np

from lib.np import *

PATH_TRAINING =  "..\\..\\datas\\training"
PATH_VALIDATION =  "..\\..\\datas\\validation"

def load(start, end):
    x_train, t_train = ([], [])
    x_test, t_test = ([], [])

    for (root, directories, files) in os.walk(PATH_TRAINING):
        for file in files:
            num = int(str(file).removesuffix(".txt"))
            if num < start or end < num:
                continue

            file_path = os.path.join(root, file)

            f = open(file_path, 'r', encoding='utf-8-sig')
            json_string = f.read()
            f.close()

            json_object = json.loads(json_string)
            x_train.append([list(json_object["input"])])
            t_train.append(list(json_object["output"]))

    for (root, directories, files) in os.walk(PATH_VALIDATION):
        for file in files:
            file_path = os.path.join(root, file)

            f = open(file_path, 'r', encoding='utf-8-sig')
            json_string = f.read()
            f.close()

            json_object = json.loads(json_string)
            x_test.append([list(json_object["input"])])
            t_test.append(list(json_object["output"]))

    x_train = np.array(x_train)
    t_train = np.array(t_train)

    x_test = np.array(x_test)
    t_test = np.array(t_test)

    return (x_train, t_train), (x_test, t_test)
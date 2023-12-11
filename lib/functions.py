from lib.np import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def cnn_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # if t.size == y.size:
    #     t = t.argmax(axis=1)

    batch_size = y.shape[0]

    # ret = 0
    #
    # for i in range(batch_size):
    #     for j in range(len(y[i])):
    #         if t[i][j] > 0 :
    #             ret += -np.log(y[i][j] + 1e-7)
    # ret /= batch_size

    bt = np.arange(batch_size)
    ret = -np.sum(np.log(y[bt][t[bt].nonzero()[0]] + 1e-7)) / batch_size

    #ret = -np.sum(np.log(y[np.arange(batch_size), t > 0] + 1e-7)) / batch_size

    return ret
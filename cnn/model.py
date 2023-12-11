import pickle

from cnn.layers import *
from lib.loss import CNNSoftmaxWithLoss

VALIDATION_BATCH_SIZE = 4

class CNN:
    def __init__(self, input_dim=(1, 1024, 128),
                 conv_param_1={'filter_num': 4, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_2={'filter_num': 4, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_3={'filter_num': 8, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_4={'filter_num': 8, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_5={'filter_num': 8, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_6={'filter_num': 8, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_7={'filter_num': 16, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_8={'filter_num': 16, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_9={'filter_num': 16, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_10={'filter_num': 16, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_11={'filter_num': 32, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_12={'filter_num': 32, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 hidden_size=256, output_size=81376):
        pre_node_nums = np.array(
            [1 * 5 * 5, 4 * 5 * 5, 4 * 5 * 5, 8 * 5 * 5, 8 * 5 * 5,8 * 5 * 5,8 * 5 * 5, 16 * 5 * 5,16 * 5 * 5,16 * 5 * 5, 16*5*5, 32*5*5, 1024, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate(
                [conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6, conv_param_7, conv_param_8, conv_param_8, conv_param_9, conv_param_10, conv_param_11, conv_param_12]):

            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
                                                                                       pre_channel_num,
                                                                                       conv_param['filter_size'],
                                                                                       conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W13'] = wight_init_scales[12] * np.random.randn(1024, hidden_size)
        self.params['b13'] = np.zeros(hidden_size)
        self.params['W14'] = wight_init_scales[13] * np.random.randn(hidden_size, output_size)
        self.params['b14'] = np.zeros(output_size)

        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W7'], self.params['b7'],
                                       conv_param_7['stride'], conv_param_7['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W8'], self.params['b8'],
                                       conv_param_8['stride'], conv_param_8['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W9'], self.params['b9'],
                                       conv_param_9['stride'], conv_param_9['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W10'], self.params['b10'],
                                       conv_param_10['stride'], conv_param_10['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W11'], self.params['b11'],
                                       conv_param_11['stride'], conv_param_11['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W12'], self.params['b12'],
                                       conv_param_12['stride'], conv_param_12['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params['W13'], self.params['b13']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W14'], self.params['b14']))
        self.layers.append(Dropout(0.5))

        self.last_layer = CNNSoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=VALIDATION_BATCH_SIZE):
        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)

            for j in range(batch_size):
                cnt = len(tt[j].nonzero()[0])
                ty = ((-1 * y[j]).argsort())
                tty = ty[:cnt]

                local_acc = 0.0
                for k in range(cnt):
                    if tt[j][int(tty[k])] > 0.0:
                        local_acc += 1
                acc += local_acc / cnt

        print(x.shape)
        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 33)):
            grads['W' + str(i + 1)] = self.layers[layer_idx].dW
            grads['b' + str(i + 1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 33)):
            self.layers[layer_idx].W = self.params['W' + str(i + 1)]
            self.layers[layer_idx].b = self.params['b' + str(i + 1)]
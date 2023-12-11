import cnn.mydata as mydata

from cnn.model import CNN
from cnn.trainer import Trainer

initial_lr = 1
for i in range(0, 16):
    (x_train, t_train), (x_test, t_test) = mydata.load(i*1024+1, i*1024+1024)
    print("[%d ~ %d번째 학습 시작]" % (1024*i+1, i*1024 + 1024))

    network = CNN()
    if i > 0:
        network.load_params("my_convnet_params-%d.pkl" % (i))

    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=5, mini_batch_size=4,
                      optimizer='Adam', optimizer_param={'lr': initial_lr},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    # 매개변수 보관
    network.save_params("my_convnet_params-%d.pkl" % (i+1))
    print("----- %d iterations -----" % (i+1))
    print("Saved Network Parameters!")

    initial_lr *= 0.8
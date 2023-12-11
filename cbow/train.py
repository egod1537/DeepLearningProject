# coding: utf-8
from lib.np import *
from lib import config
import pickle
from lib.trainer import Trainer
from lib.optimizer import Adam
from cbow.model import CBOW
from lib.util import create_contexts_target, to_cpu, to_gpu
from cbow.vocab import load_my_vocab

window_size = 5
hidden_size = 128
batch_size = 256
max_epoch = 3

corpus, word_to_id, id_to_word = load_my_vocab()
vocab_size = len(word_to_id)
print("vocab size : %d" % (vocab_size))

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)

word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params-2.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
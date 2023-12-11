import pickle
from cbow.util import most_similar

pkl_file = 'cbow_params-2.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

querys = ['']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs)
import sent2vec
from utils import *
import numpy as np

model = sent2vec.Sent2vecModel()

print('Loading model')
model.load_model('../models/embed-model-full.bin')

print('Loading tweets', end=' ')
X = load_tweets('../data/clean_test.txt')
print('(%d tweets)' % (len(X)))

print('Embedding tweets')
embs = model.embed_sentences(X)

print('Saving embeddings to out/')
np.save('../out/embeddings_test', embs)

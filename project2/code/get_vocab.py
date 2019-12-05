import sent2vec
import numpy
import pickle
model = sent2vec.Sent2vecModel()
model.load_model('../models/embed-model-full.bin') # The model can be sent2vec or cbow-c+w-ngrams
uni_embs, vocab = model.get_unigram_embeddings() # Return the full unigram embedding matrix

with open('../out/full-vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

numpy.save('../out/full-vocab-matrix', uni_embs) 

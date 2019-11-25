from utils import *
from collections import Counter
from scipy.sparse import *
import numpy as np
import pickle

# load data
X, _ = load_tweets('../data/clean_train.txt', True)

# get all unique words with their frequence
counter = Counter()
for tweet in X:
    for word in tweet.split():
        counter[word] += 1
counter = counter.most_common()
print('Found %d unique tokens' % (len(counter)))

# keep only words that appear at least `threshold` times
threshold = 5
vocab = {word: i for i,(word,count) in enumerate(counter) if count >= threshold}
print('Saving tokens that appear at least %d times (%d tokens)' % (threshold, len(vocab)))
with open('../out/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

# compute coocurence matrix
print('Computing coocurence matrix')
data, row, col = [], [], []
c = 1
for line in X:
    tokens = [vocab.get(t, -1) for t in line.strip().split()]
    tokens = [t for t in tokens if t >= 0]
    for t in tokens:
        for t2 in tokens:
            data.append(1)
            row.append(t)
            col.append(t2)

    if c % 10000 == 0:
        print('\r%.2f%%' % (c / len(X) * 100), end='')
    c += 1
cooc = coo_matrix((data, (row, col)))
print('\nSumming duplicates')
cooc.sum_duplicates()
print('Saving cooc matrix')
with open('../out/cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

# generate word embeddings with dimension `dim`
dim = 200
print('Generating %d-dimensional word embeddings' % (dim))

nmax = 100
print('Using nmax = %d, cooc.max() = %d ' % (nmax, cooc.max()))

xs = np.random.normal(size=(cooc.shape[0], dim))
ys = np.random.normal(size=(cooc.shape[1], dim))

eta = 0.001
alpha = 3 / 4

epochs = 10

for epoch in range(epochs):
    print('\rEpoch %d / %d' % (epoch+1, epochs), end='')
    for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
        logn = np.log(n)
        fn = min(1.0, (n / nmax) ** alpha)
        x, y = xs[ix, :], ys[jy, :]
        scale = 2 * eta * fn * (logn - np.dot(x, y))
        xs[ix, :] += scale * y
        ys[jy, :] += scale * x
print('\nSaving embeddings')
np.save('../out/embeddings_' + str(dim), xs)
print('Done !')

"""
all_words = set()
for tweet in X:
    for word in tweet.split():
        all_words.add(word)
vocab_length = len(all_words)
print('Vocabulary size: %d' % (vocab_length))

# create tokenizer with an extra token for unkown words
tokenizer = Tokenizer(num_words=vocab_length, oov_token=1)
tokenizer.fit_on_texts(X_train)

# tokenize data
X_train_tokenized = tokenizer.texts_to_sequences(X_train)

# get longest tweet and pad others with 0s to obtain same length
max_length = max([len(x) for x in X_train])
X_train_padded = pad_sequences(X_train_tokenized, max_length, padding='post')
print('Longest tweet: %d' % (max_length))

# create model
model = Sequential()
embedding_layer = Embedding(vocab_length + 2, 200, input_length=max_length)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# fit and evaluate the model
model.fit(X_train_padded, y_train, epochs=50, verbose=0)

# extract the embeddings
weights = emebdding_layer.get_weights()[0]
words_embeddings = {w:embeddings[idx] for w, idx in tokenizer.word_index.items()}
"""
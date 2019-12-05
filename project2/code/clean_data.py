from utils import *
from cleaning import clean_and_lemmatize
import sys

full = ''
if len(sys.argv) == 2:
    if sys.argv[1] == '-f' or sys.argv[1] == '--full':
        full = '_full'

# load training data and remove duplicates
pos = list(set(load_tweets('../data/train_pos'+ full +'.txt')))
neg = list(set(load_tweets('../data/train_neg'+ full +'.txt')))

# sort them training data to be consistent between runs (python's `set` doesn't guarantee same order on same data)
pos = sorted(pos)
neg = sorted(neg)

# clean and lemmatize the data
print('Cleaning training data ...')
cleaned_pos = [clean_and_lemmatize(t) for t in pos]
cleaned_neg = [clean_and_lemmatize(t) for t in neg]

# concatenate the data and generate labels
cleaned = cleaned_pos + cleaned_neg
labels = [1 for x in cleaned_pos] + [-1 for x in cleaned_neg]


# write out to file
print('Saving cleaned training data ...')
save_tweets('../data/clean_train'+ full +'.txt', cleaned, labels)
save_tweets('../data/clean_train'+ full +'_no_label.txt', cleaned)

print('Done.')
# ---------

# load test data
test, _ = load_tweets('../data/test_data.txt', contains_ids=True)

# clean and lemmatize
print('Cleaning test data ...')
cleaned_test = [clean_and_lemmatize(t) for t in test]

# write out to file
print('Saving cleaned test data ...')
save_tweets('../data/clean_test.txt', cleaned_test)

print('Done.')

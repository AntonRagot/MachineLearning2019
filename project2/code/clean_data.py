from utils import *
from cleaning import clean_and_lemmatize
import sys
import os.path

def clean_training_data():
    """
    Load the data from positive and negative data, clean it and save it in a new file
    """

    TRAIN_PATH_POS = '../data/train_pos_full.txt'
    TRAIN_PATH_NEG = '../data/train_neg_full.txt'
    CLEAN_TRAIN_PATH = '../data/clean/train.txt'

    if os.path.isfile(CLEAN_TRAIN_PATH):
        print('Train data already cleaned !')
        return None

    # load training data and remove duplicates
    pos = list(set(load_tweets(TRAIN_PATH_POS)))
    neg = list(set(load_tweets(TRAIN_PATH_NEG)))

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
    save_tweets(CLEAN_TRAIN_PATH, cleaned, labels)


def clean_test_data():
    TEST_PATH = '../data/test_data.txt'
    CLEAN_TEST_PATH = '../data/clean/test.txt'

    if os.path.isfile(CLEAN_TEST_PATH):
        print('Test data already cleaned !')
        return None

    # load test data
    test, _ = load_tweets(TEST_PATH, contains_ids=True)

    # clean and lemmatize
    print('Cleaning test data ...')
    cleaned_test = [clean_and_lemmatize(t) for t in test]

    # write out to file
    print('Saving cleaned test data ...')
    save_tweets(CLEAN_TEST_PATH, cleaned_test)

from utils import *
from cleaning import clean_and_lemmatize
import sys

def clean_test_data():
    # load test data
    test, _ = load_tweets('../data/test_data.txt', contains_ids=True)

    # clean and lemmatize
    print('Cleaning test data ...')
    cleaned_test = [clean_and_lemmatize(t) for t in test]

    # write out to file
    print('Saving cleaned test data ...')
    save_tweets('../data/clean_test.txt', cleaned_test)

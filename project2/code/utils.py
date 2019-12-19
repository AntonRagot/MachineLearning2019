import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_tweets(path, contains_ids=False):
    """
    Loads the data from a file. Each line must be either of form `id,tweet` or of form `tweet`.

    Args:
        path (str): Path to the file
        contains_ids (bool): If the file contains ids / labels
    Returns:
        A list containing the loaded tweets along with a list of the ids/labels if `contains_ids` was True
    """
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    if contains_ids:
        ids = [int(l.split(',')[0]) for l in lines]
        lines = [l[l.index(',')+1:] for l in lines]
        return lines, ids
    else:
        return lines

def save_tweets(path, data, labels=None):
    """
    Saves tweets (and their labels to a file).

    Args:
        path (str): Path to destination file
        data (list): The tweets to save
        labels (list): The corresponding labels (1 or -1), can be None
    """
    with open(path, 'w', encoding='utf-8') as f:
        if labels:
            for y,x in zip(labels,data):
                f.write(str(y) + ',' + x + '\n')
        else:
            for x in data:
                f.write(x + '\n')

def one_minus_one_labels_to_one_zero(labels):
    '''Converts labels from -1 (negative) and 1 (positive) to 0 and 1'''
    return [1 if y == 1 else 0 for y in labels]

def one_zero_labels_to_one_minus_one(labels):
    '''Converts labels from 0 (negative) and 1 (positive) to -1 and 1'''
    return [1 if y == 1 else -1 for y in labels]

def generate_submission(path, predictions):
    """
    Generates a csv file to be submitted to AI-Crowd.

    Args:
        path (str): Path to destination file
        predictions (list): The predictions to be saved (must be of length 10_000 and only 1 or -1)
    """
    assert len(predictions) == 10000
    print("Predicting")
    with open(path, 'w') as f:
        f.write('Id,Prediction'+'\n')
        index = 1
        for p in predictions:
            f.write(str(index)+','+str(p)+'\n')
            index += 1

    print("Done")

def predict(model):
    """
    Using the given model, predicts the smiley for each tweet of the test set

    Args:
        model (model): Model to use to predict
    """
    preds = []
    test_data = load_tweets('../data/clean/test.txt')
    longest_tweets = get_longest_tweet_size()
    print("Getting tokenizer")
    tokenizer = get_tokenizer()
    test_data = tokenizer.texts_to_sequences(test_data)
    print("Padding test tweets")
    test_data = pad_sequences(test_data, longest_tweets, padding='post')
    for l in test_data:
        preds.append(model.predict(l.reshape(1,-1)))

    preds = [1 if x > 0.5 else -1 for x in preds]
    
    return preds


def get_tokenizer():
    """
    Return the tokenizer that is fitted on the training data
    """
    X = load_tweets('../data/clean/train.txt')

    vocabulary_length = get_vocabulary_length()

    tokenizer = Tokenizer(num_words=vocabulary_length, oov_token=1)
    tokenizer.fit_on_texts(X)

    return tokenizer

def get_vocabulary_length():
    """
    Returns the size of the vocabulary
    """

    X = load_tweets('../data/clean/train.txt')

    all_word = []
    for elem in X:
        for w in elem.split(' '):
            all_word.append(w)
    all_word = list(set(all_word))
    vocab_length = len(all_word) + 13

    return vocab_length

def get_longest_tweet_size():
    X = load_tweets('../data/clean/train.txt')
    longest = max([len(x.split()) for x in X])
    return longest
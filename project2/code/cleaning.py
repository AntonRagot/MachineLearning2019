import string
import symspell
import re
from nltk.stem.wordnet import WordNetLemmatizer

def remove_tokens(tweet):
    '''Returns tweet with removed tokens such as <user> and <url>'''
    return tweet.replace('<user>', '').replace('<url>', '')

def remove_punctuation(tweet):
    '''Returns tweet without punctuation'''
    return tweet.translate(str.maketrans('', '', string.punctuation))

def replace_numbers(tweet):
    '''Replaces numbers by <number>'''
    return ' '.join(['<number>' if w.isdigit() else w for w in tweet.split()])

def replace_elong(tweet):
    '''Replaces words with repetitions by <elong>'''
    words = tweet.split()
    corrected = [re.sub(r'(\w)\1{2,}',r'\1', w) for w in words]
    out = []
    for c,w in zip(corrected,words):
        out.append(c)
        if w != c:
            out.append('<elong>')
    return ' '.join(out)

def clean_tweet(tweet):
    '''
    Cleans a given tweet by applying the following steps:
        - lowercase
        - remove tokens
        - remove punctuation
        - remove digits
    
    Args:
        tweet (str): The tweet to clean

    Returns:
        str: The cleaned tweet
    '''
    tweet = tweet.lower()
    # tweet = remove_tokens(tweet)
    tweet = remove_punctuation(tweet)
    tweet = remove_numbers(tweet)
    return tweet

def lemmatize_tweet(tweet):
    '''
    Lemmatizes the nouns and verbs of a tweet.

    Args:
        tweet (str): The tweet to lemmatize
    
    Returns:
        str: The lemmatized tweet
    '''
    lem = WordNetLemmatizer()
    words = tweet.split()
    verbs = [lem.lemmatize(w, 'v') for w in words] # verbs
    nouns = [lem.lemmatize(w, 'n') for w in verbs] # nouns
    return ' '.join(nouns)

def clean_and_lemmatize(tweet):
    '''Cleans and lemmatizes a given tweet.'''
    return lemmatize_tweet(clean_tweet(tweet))
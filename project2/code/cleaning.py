import string
import re
import pkg_resources
from nltk.stem.wordnet import WordNetLemmatizer
from symspellpy.symspellpy import SymSpell

## ----------- for hashtags ----------
# maximum edit distance per dictionary precalculation
max_edit_distance_dictionary = 0
prefix_length = 7
# create object
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
# load dictionary
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
        # term_index is the column of the term and count_index is the
        # column of the term frequency
if not sym_spell.load_dictionary(dictionary_path, term_index=0,count_index=1):
    print("Dictionary file not found")

if not sym_spell.load_bigram_dictionary(dictionary_path, term_index=0,count_index=2):
    print("Bigram dictionary file not found")

## ---------------------------------

def remove_tokens(tweet):
    '''Returns tweet with removed tokens such as <user> and <url>'''
    return tweet.replace('<user>', '').replace('<url>', '')

def remove_punctuation(tweet):
    '''Returns tweet without punctuation but keeps our tokens'''
    custom_punctuation = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~"
    tweet = tweet.translate(str.maketrans('', '', custom_punctuation))
    words = tweet.split()
    words = [w for w in words if w.isalnum() or w[0] == '<' and w[-1] == '>']
    return ' '.join(words)

def replace_numbers(tweet):
    '''Replaces numbers by <number>'''
    return ' '.join(['<number>' if w.isdigit() else w for w in tweet.split()])

def replace_elong(tweet):
    '''Replaces words with repeated letters by <elong>'''
    words = tweet.split()
    corrected = [re.sub(r'(\w)\1{2,}',r'\1', w) for w in words]
    out = []
    for c,w in zip(corrected,words):
        out.append(c)
        if w != c:
            out.append('<elong>')
    return ' '.join(out)

def replace_heart(tweet):
    '''Replaces hearts by <heart>'''
    return ' '.join(['<heart>' if '<3' in x else x for x in tweet.split(' ')])

def split_hashtag(tweet):
    '''Returns tweet that doesn't contain any hashtags'''
    words = [w for w in tweet.split() if w != '#']
    return ' '.join([sym_spell.word_segmentation(x.replace('#', '')).corrected_string if '#' in x else x for x in words])

def replace_repetition(tweet):
    ''' Changes any consecutive occurences of a word by the word and <rep> '''
    regex = r"(\b\S+)(?:\s+\1\b)+"
    
    subst = ""
    
    find = re.findall(regex, tweet)
    
    string = [x.strip() for x in re.split(regex, tweet.lower()) if x.strip() != '']
    
    for i in range(len(string)):
        string[i] = string[i].strip()
        if string[i] in find:
            string[i] += " <rep>"
        
    
    return ' '.join(string)


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

    tweet = replace_heart(tweet)
    tweet = split_hashtag(tweet)
    tweet = remove_punctuation(tweet)
    tweet = replace_repetition(tweet)
    tweet = replace_numbers(tweet)
    tweet = replace_elong(tweet)
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

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

def clean_and_lemmatize(tweet):
    '''
    Clean and lemmatize a given tweet.
    
    Arg:
        tweet (str): The tweet to clean and lemmatize

    Returns:
        str: The cleaned and lemmatized tweet
    '''
    return lemmatize_tweet(clean_tweet(tweet))

## ---------------------------------

def clean_tweet(tweet):
    '''
    Clean a given tweet by applying the following steps:
        - Lowercase (even if it is supposed to be already done)
        - Split the hashtags
        - Replace special symbols by tokens
        - Remove punctuation
        - Replace repetition by tokens
        - Replace words with repeated letters by word + token
    
    Arg:
        tweet (str): The tweet to clean

    Returns:
        str: The cleaned tweet
    '''
    tweet = tweet.lower()
    
    # The order of the calls is significant
    #tweet = split_hashtag(tweet)
    tweet = replace_exclamation(tweet)
    tweet = replace_question(tweet)
    tweet = replace_heart(tweet)
    tweet = replace_numbers(tweet)
    tweet = remove_punctuation(tweet)
    tweet = replace_repetition(tweet)
    tweet = replace_elong(tweet)
    
    return tweet

def lemmatize_tweet(tweet):
    '''
    Lemmatize the nouns and verbs of a tweet.

    Arg:
        tweet (str): The tweet to lemmatize
    
    Returns:
        str: The lemmatized tweet
    '''
    lem = WordNetLemmatizer()
    words = tweet.split()
    verbs = [lem.lemmatize(w, 'v') for w in words] # verbs
    nouns = [lem.lemmatize(w, 'n') for w in verbs] # nouns
    return ' '.join(nouns)

## ---------------------------------

def remove_tokens(tweet):
    '''Return tweet with removed tokens such as <user> and <url>.'''
    return tweet.replace('<user>', '').replace('<url>', '')

def remove_punctuation(tweet):
    '''Return tweet without punctuation but keeps our tokens'''
    custom_punctuation = "\"#$%&'()*+,-./:;=?@[\]^_`{|}~"
    tweet = tweet.translate(str.maketrans('', '', custom_punctuation))
    words = tweet.split()
    words = [w for w in words if w.isalnum() or w[0] == '<' and w[-1] == '>']
    return ' '.join(words)

def replace_exclamation(tweet):
    '''Replace "!" by <exclamation>.'''
    return ' '.join(['<exclamation>' if w == '!' else w for w in tweet.split()])

def replace_question(tweet):
    '''Replace "?" by <question>.'''
    return ' '.join(['<question>' if w == '?' else w for w in tweet.split()])

def replace_numbers(tweet):
    '''Replace numbers by <number>.'''
    return ' '.join(['<number>' if w.isdigit() else w for w in tweet.split()])

def replace_elong(tweet):
    '''Replace words with repeated letters by <elong>.'''
    words = tweet.split()
    corrected = [re.sub(r'(\w)\1{2,}',r'\1', w) for w in words]
    out = []
    for c,w in zip(corrected,words):
        out.append(c)
        if w != c:
            out.append('<elong>')
    return ' '.join(out)

def replace_heart(tweet):
    '''Replace hearts "<3" by <heart>.'''
    return ' '.join(['<heart>' if '<3' in x else x for x in tweet.split(' ')])

def split_hashtag(tweet):
    '''Return tweet that doesn't contain any hashtag, with token <hashtag>'''
    words = [w for w in tweet.split() if w != '#']
    hashtag = '<hashtag> '
    return ' '.join([hashtag.append(sym_spell.word_segmentation(x.replace('#', '')).corrected_string) if '#' in x else x for x in words])

def replace_repetition(tweet):
    '''Change any consecutive occurences of a word by the word and <rep>.'''
    
    regex = r"(\b\S+)(?:\s+\1\b)+"
    subst = ""
    
    find = re.findall(regex, tweet)
    
    string = [x.strip() for x in re.split(regex, tweet.lower()) if x.strip() != '']
    
    for i in range(len(string)):
        string[i] = string[i].strip()
        if string[i] in find:
            string[i] += " <rep>"   
    
    return ' '.join(string)
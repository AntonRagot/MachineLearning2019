from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from utils import *

# n-gram range
LO = 1
HI = 4

CLEAN_TRAIN_PATH = '../data/clean/train.txt'

def get_requested_model(model="CNN"):
    if model == "CNN":
        return generate_model_CNN()
    elif model == "CNN_GRU":
        return generate_model_CNN_GRU()
    elif model == "LR":
        return pipeline_model(model)
    elif model == "SVC":
        return pipeline_model(model)
    elif model == "TREE":
        return pipeline_model(model)
    elif model == "BAYES":
        return pipeline_model(model)
    else:
        raise ValueError("Undefined model")

def generate_model_CNN(vocabulary_size=200, Embedding_size=10, max_length=54):
    model = Sequential()
    model.add(Embedding(vocabulary_size, Embedding_size, input_length=max_length))
    model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.001),metrics=['accuracy'])
    return model

def generate_model_CNN_GRU():
    raise NotImplementedError()

def pipeline_model(model):
    if model == "LR":
        clf = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(LO, HI))),
            ('classifier', LogisticRegression(random_state=0))
        ])
    elif model == "SVC":
        clf = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(LO, HI))),
            ('classifier', LinearSVC(random_state=0))
        ])
    elif model == "TREE":
        clf = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(LO, HI))),
            ('classifier', DecisionTreeClassifier(random_state=0, max_depth=8))
        ])
    elif model == "Bayes":
        clf = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(LO, HI))),
            ('classifier', MultinomialNB(random_state=0))
        ])
    else:
        raise ValueError("Undefined model")
    train_tweets, train_labels = load_tweets(CLEAN_TRAIN_PATH, True)
    clf.fit(train_tweets, train_labels)
    return clf

def tokenize(tweet):
    return tweet.split()

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
PATH_BEST_WEIGHTS = '../model/pretrained_model_weights.hdf5'
PATH_TRAINED_WEIGHTS = '../model/trained_weights.hdf5'


def get_requested_model(model="CNN", retrain = False):
    if model == "CNN":
        return generate_model_CNN(retrain)
    elif model == "CNN_GRU":
        return generate_model_CNN_GRU(retrain)
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

def generate_model_CNN(retrain=False,vocabulary_size=200, Embedding_size=10, max_length=54):
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

    if retrain:
        print("Training model")
        checkpoint = ModelCheckpoint(PATH_TRAINED_WEIGHTS, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
        plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto',min_lr=0)
        model.fit(X,y,validation_split=0.2,epochs=100,batch_size=2048, callbacks = [checkpoint, early_stop, plateau])

        model.load_weights(PATH_TRAINED_WEIGHTS)
    else:
        print("Loading weights")
        model.load_weights(PATH_WEIGHTS)
    return model

def generate_model_CNN_GRU():
    raise NotImplementedError()

def pipeline_model(model):
    print("Training model")
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
    elif model == "BAYES":
        clf = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(LO, HI))),
            ('classifier', MultinomialNB())
        ])
    else:
        raise ValueError("Undefined model %s" % (model))
    train_tweets, train_labels = load_tweets(CLEAN_TRAIN_PATH, True)
    clf.fit(train_tweets, train_labels)
    return clf

def tokenize(tweet):
    return tweet.split()

## Import for sklean models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

## Imports for keras models
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.layers import Bidirectional, GRU, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input, concatenate
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model, load_model


from utils import *
import numpy as np
import os.path

# n-gram range
LO = 1
HI = 4

CLEAN_TRAIN_PATH = '../data/clean/train.txt'
PATH_BEST_WEIGHTS = '../model/CNN_best_weights.hdf5'
PATH_BEST_MODEL = '../model/CNN_model.h5'
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

def generate_model_CNN(retrain=False,embedding_size=200):

    #if retrain:
    vocabulary_length = get_vocabulary_length()
    longest_tweet_size = get_longest_tweet_size()

    tweet_input = Input(shape=(longest_tweet_size,), dtype='int32')
    tweet_encoder = Embedding(vocabulary_length+1, embedding_size, input_length=longest_tweet_size, embeddings_regularizer=regularizers.l2(0.01))(tweet_input)

    convs = []
    filter_sizes = [2,3,4,5]
    nb_filters = [100,200]

    for filter_size in filter_sizes:
        for nb_filter in nb_filters:
            l_conv = Conv1D(filters=nb_filter, kernel_size=filter_size, padding='valid', activation='relu', strides=1)(tweet_encoder)
            l_pool = GlobalMaxPooling1D()(l_conv)
            convs.append(l_pool)
    merged = concatenate(convs, axis=1)

    merged = Dropout(0.4)(merged)
    merged = Dense(256)(merged)
    merged = Dropout(0.4)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.4)(merged)
    merged = Dense(1)(merged)
    output = Activation('sigmoid')(merged)
    model = Model(inputs=[tweet_input], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])

    if retrain:
        print("Re training the model")

        print("Loading the train data")
        X,y = load_tweets('../data/clean/train.txt', True)
        y = np.array(list(map(lambda x: 1 if x==1 else 0, y)))

        tokenizer = get_tokenizer()
        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, get_longest_tweet_size(), padding='post')

        #Defining the callbacks
        checkpoint = ModelCheckpoint(PATH_TRAINED_WEIGHTS, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
        plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto',min_lr=0)

        #Train the model
        print("Training model")
        model.fit(X,y,validation_split=0.2,epochs=100,batch_size=2048, callbacks = [checkpoint, early_stop, plateau])

        #Load the best weights at the end of the training
        print("Loading the best weights found")
        model.load_weights(PATH_TRAINED_WEIGHTS)
    else:
        print("Loading model")
        model = load_model(PATH_BEST_MODEL)
        model.load_weights(PATH_BEST_WEIGHTS)
        
    return model

def generate_model_CNN_GRU(retrain=False,embedding_size=200):

    vocabulary_length = get_vocabulary_length()
    longest_tweet_size = get_longest_tweet_size()

    model = Sequential()
    model.add(Embedding(vocabulary_length, embedding_size, input_length=longest_tweet_size, embeddings_regularizer=regularizers.l1_l2(l1=0.02, l2=0.02)))
    model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(GRU(1024, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.001),metrics=['accuracy'])

    if retrain:
        print("Re training the model")

        print("Loading the train data")
        X,y = load_tweets('../data/clean/train.txt', True)
        y = np.array(list(map(lambda x: 1 if x==1 else 0, y)))

        tokenizer = get_tokenizer()
        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, get_longest_tweet_size(), padding='post')

        checkpoint = ModelCheckpoint(PATH_TRAINED_WEIGHTS, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
        plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto',min_lr=0)

        model.fit(X,y,validation_split=0.2,epochs=100,batch_size=2048, callbacks = [checkpoint, early_stop, plateau])

        model.load_weights(PATH_TRAINED_WEIGHTS)
    else:
        if os.path.isfile(PATH_TRAINED_WEIGHTS):
            print('No weights exists for the model, please train the data first')
            return None
        print("Loading weights")
        model.load_weights(PATH_TRAINED_WEIGHTS)

    return model

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

##Make imports
from clean_test import *
from model import *
from utils import *

PATH_WEIGHTS = ""

print("Preprocessing data")
clean_test_data()
print("Loading clean data")
X = load_tweets('../data/clean_test.txt')
print("Loading model")
model = generate_model()
print("Loading weights")
model.load_weights(PATH_WEIGHTS)
print("Predicting")
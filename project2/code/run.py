##Make imports
from clean_test import *
from model import *
from utils import *

retrain = False

if len(sys.argv) > 1:
    if sys.argv[1] == '-r' or sys.argv[1] == '--retrain':
        retrain = True

PATH_WEIGHTS = "../model/pretrained_model_weights.hdf5"

print("Preprocessing data")
clean_test_data()
print("Loading clean data")
X = load_tweets('../data/clean_test.txt')
print("Loading model")
model = get_requested_model()
if retrain:
    print("Training model")
    
else:
    print("Loading weights")
    model.load_weights(PATH_WEIGHTS)
print("Predicting")

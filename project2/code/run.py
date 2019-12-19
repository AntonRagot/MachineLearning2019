##Make imports
from clean_data import *
from model import *
from utils import *

import argparse

PATH_OUTPUT = "../output/predictions.csv"

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--retrain', action='store_true', 
    help="retrain the best NN")
parser.add_argument('-m','--model', action='store', 
    help="Choose the model to predict (from CNN, CNN_GRU, LR, SVC, TREE, BAYES)")

args = parser.parse_args()

print("Cleaning train data")
clean_training_data()
print("Loading model")
model = get_requested_model(args.model, args.retrain)
print("Cleaning test data")
clean_test_data()
print("Predicting")
predictions = predict(model)
print("Saving predictions")
generate_submission(PATH_OUTPUT, predictions)


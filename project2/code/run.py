##Make imports
import argparse

OUTPUT_PATH = "../out/predictions.csv"

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--retrain', action='store_true', default=False, 
    help="Re-train the best NN")
parser.add_argument('-m','--model', action='store', default='CNN',
    help="Choose the model to predict (from CNN, CNN_GRU, LR, SVC, TREE, BAYES)")

args = parser.parse_args()

from clean_data import *
from model import *
from utils import *

clean_training_data()

model = get_requested_model(args.model, args.retrain)

if model is not None:
    clean_test_data()

    predictions = predict(model)

    generate_submission(PATH_OUTPUT, predictions)
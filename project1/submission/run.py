# imports
import numpy as np
from proj1_helpers import *
from implementations import *
from preprocessing import *
from features import *
from constants import *
from prediction import *

# load test data

print('Loading test data')
DATA_TEST_PATH = 'data/test.csv'
_, x, ids_test = load_csv_data(DATA_TEST_PATH)

print('Loading medians')
medians = np.load('output/medians.npy')

print('Loading weights')
weights = np.load('output/weights.npy')

print('Loading optimal degrees')
degrees = np.load('output/degrees.npy')

print('Predicting')
y_pred = predict_all(x, weights, degrees, medians, UNWANTED)

OUTPUT_PATH = 'output/predictions.csv'
print('Writing out to file %s' % (OUTPUT_PATH))
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print('Done !')
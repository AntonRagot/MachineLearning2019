# imports
import numpy as np
from utils import *
from implementations import *
from preprocessing import *
from features import *
from constants import *
from prediction import *

# load test data

print('Loading test data')
_, x, ids_test = load_csv_data(DATA_TEST_PATH)

print('Loading medians')
medians = np.load(MEDIANS_PATH)

print('Loading weights')
weights = np.load(WEIGHTS_PATH)

print('Loading optimal degrees')
degrees = np.load(DEGREES_PATH)

print('Predicting')
y_pred = predict_all(x, weights, degrees, medians, UNWANTED)

print('Writing out to file %s' % (OUTPUT_PATH))
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print('Done !')
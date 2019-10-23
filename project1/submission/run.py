# imports
import numpy as np
from proj1_helpers import *
from implementations import *

# load data
print('Loading train data...')
DATA_TRAIN_PATH = 'data/train.csv'
y, X, ids = load_csv_data(DATA_TRAIN_PATH)

# 80% train + validation, 20% test
TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = split_data(X,y,train_ratio=0.8)
print('Train samples: %d | Test samples: %d' % (len(TRAIN_Y), len(TEST_Y)))

# constants
UNDEF = -999.
JET_NUM_INDEX = 22
NB_JET_VALUES = 4

unwanted0 = [4,5,6,12,22,23,24,25,26,27,28,29]
unwanted1 = [4,5,6,12,22,26,27,28,29]
unwanted2 = [22]
unwanted3 = [22]

UNWANTED = [unwanted0, unwanted1, unwanted2, unwanted3]
MEDIANS = get_medians(TRAIN_X)

K = 10
LAMBDAS = np.logspace(-8, 0, 10)
LAMBDAS = np.insert(LAMBDAS, 0, 0)
DEGREES = np.arange(11)

cc = Classifier(UNWANTED, MEDIANS, LAMBDAS, DEGREES)

# training pipeline

best_degs = []
best_lambs = []
weights = []
for i in range(NB_JET_VALUES):
    print('Starting training for jet num %d' % (i))
    x, y = cc.clean_data_for_jet_num(TRAIN_X, TRAIN_Y, i)
    _, deg, lam = cc.k_fold_cross_validation(x, y)
    best_degs.append(deg)
    best_lambs.append(lam)
    w = cc.regress(cc.expand_poly(x,deg), y, lam)
    weights.append(w)
    x, y = cc.clean_data_for_jet_num(TEST_X, TEST_Y, i)
    
print('Best degrees')
print(best_degs)
print('Best lambdas')
print(best_lambs)

print('Accuracy on test set is: %.4f' % (cc.check(TEST_X, TEST_Y, weights, best_degs)))

# load test data

print('Loading test data...')
DATA_TEST_PATH = 'data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print('Predicting')
y_pred = cc.predict_all(tX_test, weights, best_degs)

OUTPUT_PATH = 'output/predictions.csv'
print('Writing out to file %s' % (OUTPUT_PATH))
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print('Done !')
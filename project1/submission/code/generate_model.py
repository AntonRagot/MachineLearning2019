import numpy as np
from preprocessing import *
from constants import *
from features import *
from utils import *
from train import *

# load data
print('Loading train data...')
y, X, ids = load_csv_data(DATA_TRAIN_PATH)

print('Saving medians')
medians = get_medians(X)
np.save(MEDIANS_PATH, medians)

# 80% train + validation, 20% test
TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = split_data(X,y,train_ratio=0.8)
print('Train samples: %d | Test samples: %d' % (len(TRAIN_Y), len(TEST_Y)))

# training pipeline

best_degs = []
best_lambs = []
weights = []
for i in range(4):
    print('Starting training for jet num %d' % (i))
    x, y = clean_data_for_jet_num(TRAIN_X, TRAIN_Y, i, medians,UNWANTED)
    _, deg, lam = k_fold_cross_validation(x, y, DEGREES_TO_TEST, LAMBDAS_TO_TEST)
    best_degs.append(deg)
    best_lambs.append(lam)
    w = regress(add_features(x,deg), y, lam)
    weights.append(w)
    
print('Saving weights')
np.save(WEIGHTS_PATH, weights)
print('Saving best DEGREES_TO_TEST')
np.save(DEGREES_TO_TEST_PATH, best_degs)
print('Done !')
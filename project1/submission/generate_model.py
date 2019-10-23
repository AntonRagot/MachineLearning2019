import numpy as np
from preprocessing import *
from constants import *
from features import *
from proj1_helpers import *
from train import *

# load data
print('Loading train data...')
DATA_TRAIN_PATH = 'data/train.csv'
y, X, ids = load_csv_data(DATA_TRAIN_PATH)

print('Saving medians to output/medians')
medians = get_medians(X)
np.save('output/medians', medians)

# 80% train + validation, 20% test
TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = split_data(X,y,train_ratio=0.8)
print('Train samples: %d | Test samples: %d' % (len(TRAIN_Y), len(TEST_Y)))

lambdas = np.logspace(-8, 0, 10)
lambdas = np.insert(lambdas, 0, 0)
degrees = np.arange(2,11)

# training pipeline

best_degs = []
best_lambs = []
weights = []
for i in range(4):
    print('Starting training for jet num %d' % (i))
    x, y = clean_data_for_jet_num(TRAIN_X, TRAIN_Y, i, medians,UNWANTED)
    _, deg, lam = k_fold_cross_validation(x, y, degrees, lambdas)
    best_degs.append(deg)
    best_lambs.append(lam)
    w = regress(add_features(x,deg), y, lam)
    weights.append(w)
    
print('Saving weights to output/weights')
np.save('output/weights', weights)
print('Saving best degrees to output/degrees')
np.save('output/degrees', best_degs)
print(best_degs)
print('Best lambdas')
print(best_lambs)
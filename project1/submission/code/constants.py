import numpy as np

DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'
MEDIANS_PATH = '../output/medians.npy'
WEIGHTS_PATH = '../output/weights.npy'
DEGREES_PATH = '../output/degrees.npy'
OUTPUT_PATH = '../output/predictions.csv'

UNDEF = -999.
JET_NUM_INDEX = 22
K = 10

UNWANTED = [[4,5,6,12,22,23,24,25,26,27,28,29],
            [4,5,6,12,22,26,27,28,29],
            [22],
            [22]]
            
LAMBDAS_TO_TEST = np.logspace(-8, 0, 10)
LAMBDAS_TO_TEST = np.insert(LAMBDAS_TO_TEST, 0, 0)
DEGREES_TO_TEST = np.arange(2,11)
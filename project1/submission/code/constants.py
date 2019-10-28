import numpy as np

# file paths
DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'
MEDIANS_PATH = '../output/medians.npy'
WEIGHTS_PATH = '../output/weights.npy'
DEGREES_PATH = '../output/degrees.npy'
OUTPUT_PATH = '../output/predictions.csv'

# global constants
UNDEF = -999.
JET_NUM_INDEX = 22

# training constants
K = 10
UNWANTED = [[4,5,6,12,22,23,24,25,26,27,28,29],
            [4,5,6,12,22,26,27,28,29],
            [22],
            [22]]
LAMBDAS_TO_TEST = np.hstack(([0], np.logspace(-8, 0, 20)))
DEGREES_TO_TEST = np.arange(4, 11)
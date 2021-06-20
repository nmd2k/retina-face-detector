# DATA config ================
DATA_PATH           = './data'
N_CLASSES           = 2
TRAIN_PATH          = './data/train'
VALID_PATH          = './data/val'
TEST_PATH           = './data/test'
SAVE_PATH           = './model'

DATASET_NAME        = 'widerface' 
DATASET_VER         = ':latest'
BATCH_SIZE          = 32
RANDOM_SEED         = 42
VALID_RATIO         = 0.2
NUM_WORKERS         = 0

# MODEL config ==============
EPOCHS              = 5
START_FRAME         = 8
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 5e-4
MOMENTUM            = 0.9


# CRITERION config ==========
OVERLAP_THRES       = 0.35
BKG_LABEL           = 0
NEG_MINING          = 7
NEG_OVERLAP         = 0.35

# DATA config ================
DATA_PATH           = './data'
N_CLASSES           = 2
TRAIN_PATH          = './train'
VALID_PATH          = './val'
TEST_PATH           = './test'
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
# FEATURE_MAP         = {'stage1': 80, 'stage2': 40, 'stage3': 20}
FEATURE_MAP         = {'stage2': 160, 'stage3': 80, 'stage4':40, 'stage5':20, 'stage6':10}
RETURN_LAYERS       = {'stage1': 1, 'stage2': 2, 'stage3': 3}
IN_CHANNELS         = 32
OUT_CHANNELS        = 64

# CRITERION config ==========
OVERLAP_THRES       = 0.35
BKG_LABEL           = 0
NEG_MINING          = 7
NEG_OVERLAP         = 0.35

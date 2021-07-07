PROJECT             = 'Retina-Face'
RUN_NAME            = 'resnet18'
# DATA config ================
DATA_PATH           = './data'
N_CLASSES           = 2
TRAIN_PATH          = './train'
VALID_PATH          = './val'
TEST_PATH           = './test'
SAVE_PATH           = './model'

DATASET             = 'wider-face-4k' 
DVERSION            = 'latest'
INPUT_SIZE          = 640
BATCH_SIZE          = 8
RANDOM_SEED         = 42
NUM_WORKERS         = 0

# MODEL config ==============
EPOCHS              = 5
START_FRAME         = 8
LEARNING_RATE       = 1e-3
LR_MILESTONE        = [4,9,14,19]
WEIGHT_DECAY        = 5e-4
MOMENTUM            = 0.9

IN_CHANNELS         = 32
OUT_CHANNELS        = 64

# MobileNetV1
# FEATURE_MAP_MOBN1   =  {'layer2': 160,
#                         'layer3': 80,
#                         'layer4': 40,
#                         'layer5': 20,}
FEATURE_MAP_MOBN1   = [2, 3, 4, 5]
RETURN_MAP_MOBN1    =  {'layer2': 'out_feature2', 
                        'layer3': 'out_feature3', 
                        'layer4': 'out_feature4'}

# MobileNetV2
FEATURE_MAP_MOBN2   = {}
RETURN_LAYERS       = {'layer1': 1, 'layer2': 2, 'layer3': 3}
# FEATURE_MAP         = {'layer2': 160, 'layer3': 80, 'layer4':40, 'layer5':20, 'layer6':10}

# Resnet50
FEATURE_MAP         = [2, 3, 4, 5, 6]
RETURN_MAP          = {'layer1': 'out_feature1',
                        'layer2': 'out_feature2', 
                        'layer3': 'out_feature3',
                        'layer4': 'out_feature4'}

# CRITERION config ==========
OVERLAP_THRES       = 0.35
BKG_LABEL           = 0
NEG_MINING          = 7
NEG_OVERLAP         = 0.35

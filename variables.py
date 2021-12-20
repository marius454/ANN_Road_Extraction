# The Shape of the model and the resizing of images (when evaluating a saved model these must be set to the same size)
img_width = 256   #224 max
img_height = 256  #224 max
# The path to the training and tesing data (data is only extranted no other processing is done outside of code)
TRAIN_PATH = "../data/train/"
TEST_PATH = "../data/test/"
# The number of samples taken for training 
TRAIN_LENGTH = 3000 # 5292 max
TEST_LENGTH = 934   # 934 max
BATCH_SIZE = 16
# Used for shuffle, best practice is the same size or larger than the dataset
BUFFER_SIZE = TRAIN_LENGTH
EPOCHS = 30
# STEPS_PER_EPOCH = int(round(TRAIN_LENGTH / BATCH_SIZE))
STEPS_PER_EPOCH = int(TRAIN_LENGTH / BATCH_SIZE)
# Class weights for background and road pixels
BACKGROUND_WEIGHT = 1 
ROAD_WEIGHT = 3

# Geriausias variantas svoriai 1:3 ir 15 epochu, nes veliau prasideda overfitting

# Some global variables
model = None
sample_image = None
sample_mask = None
sample_weight = list()
# The Shape of the model and the resizing of images
img_width = 128   #224 max
img_height = 128  #224 max

# The path to the training and tesing data (data is only extranted no other processing is done outside of code)
TRAIN_PATH = "../data/train/"
TEST_PATH = "../data/test/"
# The number of samples taken for training 
TRAIN_LENGTH = 1000 # 5292 max
TEST_LENGTH = 150   # 934 max
BATCH_SIZE = 64
# Used for shuffle, best practice is the same size or larger than the dataset
BUFFER_SIZE = TRAIN_LENGTH
EPOCHS = 10
# Class weights for background and road pixels
BACKGROUND_WEIGHT = 1
ROAD_WEIGHT = 8

# Some global variables
model = None
sample_image = None
sample_mask = None
sample_weight = list()
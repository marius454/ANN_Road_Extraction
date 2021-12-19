# The Shape of the model and the resizing of images (when evaluating a saved model these must be set to the same size)
img_width = 224   #224 max
img_height = 224  #224 max
# The path to the training and tesing data (data is only extranted no other processing is done outside of code)
TRAIN_PATH = "../data/train/"
TEST_PATH = "../data/test/"
# The number of samples taken for training 
TRAIN_LENGTH = 5292 # 5292 max
TEST_LENGTH = 100   # 934 max
BATCH_SIZE = 16
# Used for shuffle, best practice is the same size or larger than the dataset
BUFFER_SIZE = TRAIN_LENGTH
EPOCHS = 20
# Class weights for background and road pixels
BACKGROUND_WEIGHT = 1 
ROAD_WEIGHT = 5 

# Some global variables
model = None
sample_image = None
sample_mask = None
sample_weight = list()
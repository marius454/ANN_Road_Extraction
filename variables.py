img_width = 128   #224
img_height = 128  #224

TRAIN_LENGTH = 2000
TEST_LENGTH = 300
BATCH_SIZE = 64
# Used for shuffle best practice is for it to be the same size or larger than the dataset
BUFFER_SIZE = TRAIN_LENGTH
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

EPOCHS = 40
VAL_SUBSPLITS = 5
VALIDATION_STEPS = TEST_LENGTH//BATCH_SIZE//VAL_SUBSPLITS

model = None
sample_image = None
sample_mask = None
sample_weight = list()
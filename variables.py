img_width = 224
img_height = 224

TRAIN_LENGTH = 200
BATCH_SIZE = 32
# Used for shuffle best practice is for it to be the same size or larger than the dataset
BUFFER_SIZE = TRAIN_LENGTH
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
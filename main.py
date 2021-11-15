import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import time

from Augment import Augment
import file_processing as fp
from display import display
import variables as var
import model


# train_batches = (
#     fp.load_data("../data/train(small)/")
#     .cache()
#     .shuffle(var.BUFFER_SIZE)
#     .batch(var.BATCH_SIZE)
#     .repeat()
#     .map(Augment())
#     .prefetch(buffer_size=tf.data.AUTOTUNE))

# test_batches = fp.load_data("../data/test(small)/").batch(var.BATCH_SIZE)


# for images, masks in train_batches.take(3):
#   sample_image, sample_mask = images[0], masks[0]
#   display([sample_image, sample_mask])

model.create_model()


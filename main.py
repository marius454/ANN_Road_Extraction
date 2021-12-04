import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import time

from Augment import Augment
import file_processing as fp
from display import display
import variables as var
import Model
from DisplayCallback import DisplayCallback


train_batches = (
    fp.load_data("../data/train/", True)
    .cache()
    .shuffle(var.BUFFER_SIZE)
    .batch(var.BATCH_SIZE)
    # .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = fp.load_data("../data/validation/", False).batch(var.BATCH_SIZE)

for images, masks in train_batches.take(1):
  var.sample_image, var.sample_mask = images[0], masks[0]
#   display([var.sample_image, var.sample_mask])

# Model.create_model(visualize=True)
# Model.show_predictions()

var.model = Model.create_model()
# print(var.sample_weight[0])

model_history = var.model.fit(train_batches, epochs=var.EPOCHS,
                          # steps_per_epoch=var.STEPS_PER_EPOCH - 1,
                          # validation_steps=5,
                          steps_per_epoch = None,
                          validation_steps = None,
                          # sample_weight = var.sample_weight,
                          # class_weight = { 0: 0.05, 1: 0.95},
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()],
                          verbose = 1)


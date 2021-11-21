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
    fp.load_data("../data/train(small)/")
    .cache()
    .shuffle(var.BUFFER_SIZE)
    .batch(var.BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = fp.load_data("../data/test(small)/").batch(var.BATCH_SIZE)

# for images, masks in train_batches.take(1):
#   var.sample_image, var.sample_mask = images[0], masks[0]
#   # display([sample_image, sample_mask])

# # Model.create_model(visualize=True)
# # Model.show_predictions()

# var.model = Model.create_model()

# model_history = var.model.fit(train_batches, epochs=var.EPOCHS,
#                           steps_per_epoch=var.STEPS_PER_EPOCH,
#                           validation_steps=var.VALIDATION_STEPS,
#                           validation_data=test_batches,
#                           callbacks=[DisplayCallback()])


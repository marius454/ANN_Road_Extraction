import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import time
import string

from Augment import Augment
import file_processing as fp
from display import display
import variables as var
import Model
from callbacks import DisplayCallback

checkpoint_path = ""
checkpoint_name = "brute_force"
if tf.test.is_gpu_available():
  checkpoint_path = "training/" + tf.test.gpu_device_name().translate(str.maketrans('','',string.punctuation)) + checkpoint_name
else:
  checkpoint_path = "training/" + checkpoint_name
chekcpoint_dir = os.path.dirname(checkpoint_path)

def train_model():
  train_batches = (
      fp.load_data(var.TRAIN_PATH, True)
      .cache()
      .shuffle(var.BUFFER_SIZE)
      .batch(var.BATCH_SIZE)
      # .repeat()
      .map(Augment())
      .prefetch(buffer_size=tf.data.AUTOTUNE))

  test_batches = fp.load_data(var.TEST_PATH, False).batch(var.BATCH_SIZE)

  for images, masks in train_batches.take(3):
    var.sample_image, var.sample_mask = images[0], masks[0]
    # var.sample_images.append(images[0])
    # var.sample_masks.append(masks[0])
  #   display([var.sample_image, var.sample_mask])

  # Model.create_model(visualize=True)
  # Model.show_predictions()

  var.model = Model.create_model()

  SaveCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  # save_freq=int(5*round(var.TRAIN_LENGTH / var.BATCH_SIZE)),
                                                  verbose=1)

  model_history = var.model.fit(train_batches, epochs=var.EPOCHS,
                            steps_per_epoch = None,
                            validation_steps = None,
                            validation_data=test_batches,
                            # callbacks=[DisplayCallback()],
                            callbacks=[DisplayCallback(), SaveCallback],
                            verbose = 1)

  # Model.show_predictions(test_batches)

def load_trained_model():
  var.model = Model.create_model()
  var.model.load_weights(checkpoint_path)

  test_batches = fp.load_data(var.TEST_PATH, False, num = 3).batch(var.BATCH_SIZE)

  Model.show_predictions(test_batches)

# Leave only one of these uncommented at one time
# Use for loading an already trained model, change "checkpoint_name" variable to select which model to run
load_trained_model()

# Use to train a model. Warning: might overwrite previous model, if this is unwanted,
# change the "checkpoint_name" variable.
# train_model()

  


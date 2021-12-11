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

# Set paths for loading or saving models
checkpoint_path = ""
checkpoint_name = "test"
if tf.test.is_gpu_available():
  checkpoint_path = "training/" + tf.test.gpu_device_name().translate(str.maketrans('','',string.punctuation)) + checkpoint_name
else:
  checkpoint_path = "training/" + checkpoint_name
chekcpoint_dir = os.path.dirname(checkpoint_path)

# Train model based on the variables given in the variables.py file
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


def load_trained_model(save_path, num_samples):
  var.model = Model.create_model()
  var.model.load_weights(save_path)

  test_data = fp.load_data(var.TEST_PATH, training = False, num = num_samples).batch(1)
  train_data = fp.load_data(var.TRAIN_PATH, training = True, num = num_samples).batch(1)

  Model.show_predictions(test_data)
  Model.show_predictions(train_data)

# Leave only one of these uncommented at one time
# Use for loading an already trained model and displaying the first 3 image predictions from testing data and 3 image predictions from training data
load_trained_model("training/deviceGPU040_epochs", 3) # trained with GPU all data and 40 epochs
# load_trained_model("training/deviceGPU0brute_force", 3) # trained with GPU all data and 80 epochs

# Use to train a model. Warning: might overwrite previous model, if this is unwanted,
# change the "checkpoint_name" variable.
# train_model()

  


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
import model
from callbacks import DisplayCallback, KaggleExampleDisplayCallback

# Set paths for loading or saving models
checkpoint_path = ""
# checkpoint_name = "_Weight1-3_Epochs30-best_Image224_Data3000_shuffle_repeat_batchnorm"
checkpoint_name = "_Kaggle_comparison_data3000"
if tf.test.is_gpu_available():
  checkpoint_path = "training/" + tf.test.gpu_device_name().translate(str.maketrans('','',string.punctuation)) + checkpoint_name
else:
  checkpoint_path = "training/" + checkpoint_name
chekcpoint_dir = os.path.dirname(checkpoint_path)


# Train model based on the variables given in the variables.py file
def train_model():
  # Load training data
  startTime = time.time()
  # tf.keras.mixed_precision.set_global_policy('float16')
  train_batches = (
      fp.load_data(var.TRAIN_PATH, True)
      .cache()
      .shuffle(var.BUFFER_SIZE)
      .batch(var.BATCH_SIZE)
      # .repeat()
      .map(Augment())
      .prefetch(buffer_size=tf.data.AUTOTUNE))
  # Load_testing data
  test_batches = fp.load_data(var.TEST_PATH, False).batch(var.BATCH_SIZE)
  endTime = time.time()
  print (endTime - startTime)

  print (train_batches)

  # Select a sample image and mask for display callback
  for images, masks in train_batches.take(4):
    var.sample_image, var.sample_mask = images[0], masks[0]
  # Add class weights to training data
  # train_batches = train_batches.map(fp.add_sample_weights)
  # test_batches = test_batches.map(fp.add_sample_weights)

  # Instantiate model
  var.model = model.create_model()

  # Instantiate callback for saving model to file
  SaveCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  save_best_only = True,
                                                  monitor = "val_IoU",
                                                  mode = "max",
                                                  # save_freq=5*var.STEPS_PER_EPOCH + 1,
                                                  verbose=1)
  ## Train the model
  model_history = var.model.fit(train_batches, epochs=var.EPOCHS,
                            # steps_per_epoch = var.STEPS_PER_EPOCH,
                            steps_per_epoch = None,
                            validation_steps = None,
                            validation_data=test_batches,
                            callbacks=[DisplayCallback(), SaveCallback],
                            verbose = 1,
                            shuffle = True
                            )

  ## Train kaggle model
  # model_history = var.model.fit(train_batches, epochs=var.EPOCHS,
  #                           steps_per_epoch = None,
  #                           validation_steps = None,
  #                           validation_data=test_batches,
  #                           callbacks=[KaggleExampleDisplayCallback(), SaveCallback],
  #                           verbose = 1,
  #                           )


def load_trained_model(save_path, num_test, num_train, evaluate=False):
  # tf.keras.mixed_precision.set_global_policy('float16')
  var.model = model.create_model()
  var.model.load_weights(save_path)

  if (evaluate):
    # Evaluate trained model
    train_batches = (
      fp.load_data(var.TRAIN_PATH, True, num = num_train)
      .cache()
      .shuffle(var.BUFFER_SIZE)
      .batch(var.BATCH_SIZE)
      .repeat()
      .map(Augment())
      .prefetch(buffer_size=tf.data.AUTOTUNE))
    test_batches = fp.load_data(var.TEST_PATH, False, num = num_test).batch(var.BATCH_SIZE)
    train_batches = train_batches.map(fp.add_sample_weights)
    test_batches = test_batches.map(fp.add_sample_weights)

    print ("background pixel to road pixel class weight ratio: " + str(var.BACKGROUND_WEIGHT) +":" + str(var.ROAD_WEIGHT))
    loss, acc, iou = var.model.evaluate(train_batches, verbose=1)
    print("Trained model accuracy on training data: {:5.2f}%".format(100 * acc))
    loss, acc, iou = var.model.evaluate(test_batches, verbose=1)
    print("Trained model accuracy on test data: {:5.2f}%".format(100 * acc))
  else:
    # Show sample prediction based on saved model
    test_data = fp.load_data(var.TEST_PATH, training = False, num = num_test).batch(1)
    train_data = fp.load_data(var.TRAIN_PATH, training = True, num = num_train).batch(1)
    model.show_predictions(test_data)
    model.show_predictions(train_data)

## Leave only one of these uncommented at one time
## Use for loading an already trained model and displaying the set number of images and their coresponding predictions
## num_test selects the number of images to show from the test data, num_train does the same for training data
# load_trained_model("training/deviceGPU040_epochs", num_test=10, num_train=1) # trained with GPU all data and 40 epochs
# load_trained_model("training/deviceGPU0_Batch16_Sample1-6_Epochs20_Image128", num_test=10, num_train=1) # trained with GPU all data, batch size of 16, class weight ratio in 
# load_trained_model("training/deviceGPU0_Batch16_Sample1-6_Epochs20_Image224", num_test=10, num_train=1) # trained with GPU all data and 40 epochs

load_trained_model("training/deviceGPU0_Weight1-3_Epochs30-best_Image224_Data3000", num_test=10, num_train=1)

## Use for loading the selected number of testing and training data and evaluate the selected model on that data
# load_trained_model("training/deviceGPU0_Weight1-3_Epochs30-best_Image224_Data3000", num_test = 934, num_train = 100, evaluate=True)

## Use to train a model. Warning: might overwrite previous model, if this is unwanted, change the "checkpoint_name" variable.
## You can set the variables for training in the variables.py file
# train_model()
# Geriausias variantas svoriai 1:3 ir 15 epochu, nes veliau prasideda overfitting


## function used to evaluate 6 models at the same time, for this to work all selected model should be trained with image resolution equal to the one set in variables.tf file
## Also, it is not necesary for the batch size in the first line of the code to match the batch size used for model training.
## Uncoment the line right under this function to use it
def evaluate_multiple():
  var.BATCH_SIZE = 32
  #Load testing data
  test_batches = fp.load_data(var.TEST_PATH, False).batch(var.BATCH_SIZE)

  # 1 model:
  # Set saved model path:
  save_path = "training/deviceGPU0_Batch32_Sample1-6_Epochs20_Image224"
  # Load model
  model1 = model.create_model()
  model1.load_weights(save_path)
  print(model1.summary)
  # Evaluate model
  print (save_path)
  loss, acc, iou = model1.evaluate(test_batches, verbose=1)


  # 2 model:
  # set saved model path:
  save_path = "training/deviceGPU0_Batch32_Sample1-8_Epochs20_Image224"
  # Load model
  model1 = model.create_model(visualize=True)
  model1.load_weights(save_path)
  # Evaluate model
  print (save_path)
  loss, acc, iou = model1.evaluate(test_batches, verbose=1)

  # 3 model:
  # set saved model path:
  save_path = "training/deviceGPU0_Batch32_Sample1-10_Epochs20_Image224"
  # Load model
  model1 = model.create_model()
  model1.load_weights(save_path)
  # Evaluate model
  print (save_path)
  loss, acc, iou = model1.evaluate(test_batches, verbose=1)

  # 4 model:
  # set saved model path:
  save_path = "training/deviceGPU0_Batch32_Sample1-6_Epochs40_Image224"
  # Load model
  model1 = model.create_model()
  model1.load_weights(save_path)
  # Evaluate model
  print (save_path)
  loss, acc, iou = model1.evaluate(test_batches, verbose=1)

  # 5 model:
  # set saved model path:
  save_path = "training/deviceGPU0_Batch32_Sample1-8_Epochs40_Image224"
  # Load model
  model1 = model.create_model()
  model1.load_weights(save_path)
  # Evaluate model
  print (save_path)
  loss, acc, iou = model1.evaluate(test_batches, verbose=1)

  # 6 model:
  # set saved model path:
  save_path = "training/deviceGPU0_Batch32_Sample1-10_Epochs40_Image224"
  # Load model
  model1 = model.create_model()
  model1.load_weights(save_path)
  # Evaluate model
  print (save_path)
  loss, acc, iou = model1.evaluate(test_batches, verbose=1)
  

# evaluate_multiple()
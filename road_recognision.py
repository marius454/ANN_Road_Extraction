import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
import time
import matplotlib.pyplot as plt

# dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

# print(type(dataset["train"]))

img_width = 1024
img_height = 1024

TRAIN_LENGTH = 200
BATCH_SIZE = 20
# Used for shuffle? best practice is for it to be the same size as the dataset
BUFFER_SIZE = TRAIN_LENGTH
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


# Normalize image color values to range [0, 1] in stead of [0-255]
def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

# Load images from selected directory into a tensorflow dataset
def load_data (directory):
    images = []
    masks = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = tf.io.read_file(directory + filename,)
            img = decode_img(img)
            img = normalize(img)
            images.append (img)
        else:
            mask = tf.io.read_file(directory + filename)
            mask = decode_mask(mask)
            mask = normalize(mask)
            mask = tf.image.convert_image_dtype(mask, tf.uint8)
            masks.append(mask)

    images = tf.convert_to_tensor(images)
    masks = tf.convert_to_tensor(masks)


    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    print(dataset)
    print(type(dataset))

    return dataset

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])

def decode_mask(mask):
    mask = tf.io.decode_png(mask, channels=1)
    return tf.image.resize(mask, [img_height, img_width])


class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


# train_images = load_data("../data/train(small)/")
# test_images = load_data("../data/test(small)/")

train_batches = (
    load_data("../data/train(small)/")
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = load_data("../data/test(small)/").batch(BATCH_SIZE)


for images, masks in train_batches.take(3):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])


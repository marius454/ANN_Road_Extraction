import tensorflow as tf
import os
import variables as var
from time import sleep
import numpy as np
from random import randrange


# Normalize image color values to range [0, 1] in stead of [0-255]
def normalize_image(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

# Normalize(threshold) the mask to a binary 0 or 1 mask
def normalize_mask(input_mask):
    input_mask = tf.where(input_mask > 1, 1, 0)
    return input_mask

def load_data (directory, training, num = None):
    print ("starting to load data from directory " + directory)
    images = []
    masks = []

    # Choose the number of images to take from the folder
    if num:
        search_length = num
    else:
        if training:
            search_length = var.TRAIN_LENGTH
        else:
            search_length = var.TEST_LENGTH

    # Load data from folder into memory, search_length is doubled, because there are 2 images for each sample (image and mask)
    for i in range(search_length * 2):
        filename = os.listdir(directory)[i]
        if filename.endswith(".jpg"):
            img = tf.io.read_file(directory + filename)
            img = decode_img(img)
            img = normalize_image(img)
            images.append (img)
        else:
            mask = tf.io.read_file(directory + filename)
            mask = decode_mask(mask)
            mask = normalize_mask(mask)

            masks.append(mask)

    # Transform sets of images to tensors
    images = tf.convert_to_tensor(images)
    masks = tf.convert_to_tensor(masks)

    print ("loading complete " + directory)
    # Create and return a tensorflow dataset
    return tf.data.Dataset.from_tensor_slices((images, masks))

# Decode 3 channel jpeg image
def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [var.img_height, var.img_width])

# Decode 1 channel png image (for masks)
def decode_mask(mask):
    mask = tf.io.decode_png(mask, channels=1)
    return tf.image.resize(mask, [var.img_height, var.img_width])

def add_sample_weights(image, label):
    # Set class weights
    class_weights = tf.constant([var.BACKGROUND_WEIGHT, var.ROAD_WEIGHT])
    # Normalize to range [0, 1]
    class_weights = tf.cast((class_weights/tf.reduce_sum(class_weights)), dtype=tf.float32)
    # Create sample weights
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights
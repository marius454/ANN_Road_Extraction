import tensorflow as tf
import os
import variables as var
from time import sleep
import numpy as np

from sklearn.utils.class_weight import compute_sample_weight


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

    if num:
        search_length = num
    else:
        if training:
            search_length = var.TRAIN_LENGTH
        else:
            search_length = var.TEST_LENGTH

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

    images = tf.convert_to_tensor(images)
    masks = tf.convert_to_tensor(masks)

    print ("loading complete " + directory)
    return tf.data.Dataset.from_tensor_slices((images, masks))

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [var.img_height, var.img_width])

def decode_mask(mask):
    mask = tf.io.decode_png(mask, channels=1)
    return tf.image.resize(mask, [var.img_height, var.img_width])
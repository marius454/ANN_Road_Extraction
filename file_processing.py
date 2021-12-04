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

def normalize_mask(input_mask):
    input_mask = tf.where(input_mask > 128, 1, 0)
    return input_mask
    # for i in len(input_mask):
    #     if input_mask[i] < 128:
    #         input_mask[i] = 0
    #     else:
    #         input_mask[i] = 1
    # return input_mask

# Load images from selected directory into a tensorflow dataset
def load_data (directory, training):
    images = []
    masks = []
    sample_weight = []
    serach_length = 0
    if training:
        search_length = var.TRAIN_LENGTH
    else:
        search_length = var.TEST_LENGTH

    # for filename in os.listdir(directory):
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
            # mask = normalize_image(mask)
            mask = normalize_mask(mask)

            # list_mask = mask.numpy().tolist()
            # flat_list_mask = [item for sublist in list_mask for item in sublist]
            # flat_flat_list_mask = [item for sublist in flat_list_mask for item in sublist]
            # sample_weight.append(flat_flat_list_mask)
            masks.append(mask)

    # print(var.sample_weight[0])
    images = tf.convert_to_tensor(images)
    masks = tf.convert_to_tensor(masks)

    # var.sample_weight = compute_sample_weight(class_weight='balanced', y = sample_weight)

    # dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    # dataset = tf.data.Dataset.from_tensor_slices((images, masks, var.sample_weight))

    return tf.data.Dataset.from_tensor_slices((images, masks))

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [var.img_height, var.img_width])

def decode_mask(mask):
    mask = tf.io.decode_png(mask, channels=1)
    return tf.image.resize(mask, [var.img_height, var.img_width])
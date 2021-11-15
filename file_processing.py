import tensorflow as tf
import os
import variables as var
from time import sleep


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
    sleep(10)
    print(dataset)
    print(type(dataset))

    return dataset

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [var.img_height, var.img_width])

def decode_mask(mask):
    mask = tf.io.decode_png(mask, channels=1)
    return tf.image.resize(mask, [var.img_height, var.img_width])
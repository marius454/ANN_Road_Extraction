import tensorflow as tf
import os
import variables as var
from time import sleep


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
def load_data (directory):
    images = []
    masks = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = tf.io.read_file(directory + filename,)
            img = decode_img(img)
            img = normalize_image(img)
            images.append (img)
        else:
            mask = tf.io.read_file(directory + filename)
            mask = decode_mask(mask)
            mask = normalize_image(mask)
            # mask = tf.image.convert_image_dtype(mask, tf.uint8)
            masks.append(mask)

    images = tf.convert_to_tensor(images)
    masks = tf.convert_to_tensor(masks)
    print (images[0])
    print (masks[0])


    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    print(dataset)
    # print(dataset)
    # print(type(dataset))

    return dataset

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [var.img_height, var.img_width])

def decode_mask(mask):
    mask = tf.io.decode_png(mask, channels=1)
    return tf.image.resize(mask, [var.img_height, var.img_width])
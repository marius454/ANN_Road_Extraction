import tensorflow as tf
import variables as var
from tensorflow_examples.models.pix2pix import pix2pix
from display import display#, display3

base_model = tf.keras.applications.MobileNetV2(input_shape=[var.img_width, var.img_height, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project',
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(var.img_width * 4, 3),
    pix2pix.upsample(var.img_width * 2, 3),
    pix2pix.upsample(var.img_width, 3),
    pix2pix.upsample(var.img_width / 2, 3),
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[var.img_width, var.img_height, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same') 

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def create_model(visualize=False):
    OUTPUT_CLASSES = 2

    model = unet_model(output_channels=OUTPUT_CLASSES)
    model.compile(optimizer='adam',
                # loss = "binary_crossentropy",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Set visualize to true if you want to change the image in Model.png
    if (visualize):
      tf.keras.utils.plot_model(model, show_shapes=True)

    return model

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset:
      pred_mask = var.model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([var.sample_image, var.sample_mask, create_mask(var.model.predict(var.sample_image[tf.newaxis, ...]))])

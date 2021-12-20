from numpy import dtype
import tensorflow as tf
import variables as var
from tensorflow_examples.models.pix2pix import pix2pix
from display import display

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

  ## dropout layer to combar overfitting
  # dropout_layer = tf.keras.layers.Dropout(rate = 0.1)
  # x = dropout_layer(inputs)

  # Downsampling through the model
  skips = down_stack(inputs)
  # skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  cnn_layer = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same') 
  x = cnn_layer(x)

  # Softmax layer
  # x = tf.keras.layers.Softmax()(x)

  # Batch normalization
  # x = tf.keras.layers.BatchNormalization()(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def create_model(visualize=False):
    OUTPUT_CLASSES = 2

    ## Original model initialization
    # iou = UpdatedIoU(
    #   num_classes = OUTPUT_CLASSES,
    #   target_class_ids = [1],
    #   name = "IoU"
    # )
    # model = unet_model(output_channels=OUTPUT_CLASSES)
    # model.compile(optimizer='adam',
    #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #             metrics=['accuracy', iou],
    #           )
    ## -----------------------------

    ## Initialize different model from kaggle:
    iou = KaggleIoU(
      num_classes = OUTPUT_CLASSES,
      target_class_ids = [1],
      name = "IoU"
    )

    inputs = tf.keras.layers.Input((var.img_width, var.img_width, 3))
    model = GiveMeUnet(inputs, droupouts= 0.07)
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy', iou] )
    
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='UnetArchitecture.png', show_shapes=True, show_layer_names=True)
    ## -----------------------------

    # Set visualize to true if you want to change the image in Model.png
    if (visualize):
      tf.keras.utils.plot_model(model, show_shapes=True)

    return model

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def create_range_mask(pred_mask):
  pred_mask = tf.unstack(pred_mask, axis=-1)[1]
  return pred_mask[0]

def show_predictions(dataset=None, kaggle = False):
  if dataset:
    for image, mask in dataset:
      pred_mask = var.model.predict(image)
      if kaggle:
        display([image[0], mask[0], pred_mask[0]])
      else:  
        display([image[0], mask[0], create_range_mask(pred_mask)])
  else:
    if kaggle:
      display([var.sample_image, var.sample_mask, var.model.predict(var.sample_image[tf.newaxis, ...])[0]])
    else:  
      display([var.sample_image, var.sample_mask, create_range_mask(var.model.predict(var.sample_image[tf.newaxis, ...]))])
    
# A modified IoU metric class for compatibility with SparseCategoricalCrossentropy() loss function
class UpdatedIoU(tf.keras.metrics.IoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               target_class_ids = None,
               name=None,
               dtype=None):
    super(UpdatedIoU, self).__init__(num_classes = num_classes, target_class_ids = target_class_ids, name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)


def normalize_pred(input):
    input = tf.where(input > 0.5, 1, 0)
    input = tf.cast(input, tf.int16)
    return input

class KaggleIoU(tf.keras.metrics.IoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               target_class_ids = None,
               name=None,
               dtype=None):
    super(KaggleIoU, self).__init__(num_classes = num_classes, target_class_ids = target_class_ids, name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = normalize_pred(y_pred)
    return super().update_state(y_true, y_pred, sample_weight)





## Different model form Kaggle:
# defining Conv2d block for our u-net
# this block essentially performs 2 convolution

def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
    #first Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (inputTensor)
    
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x =tf.keras.layers.Activation('relu')(x)
    
    #Second Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation('relu')(x)
    
    return x



# Now defining Unet 
def GiveMeUnet(inputImage, numFilters = 16, droupouts = 0.1, doBatchNorm = True):
    # defining encoder Path
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    p1 = tf.keras.layers.Dropout(droupouts)(p1)
    
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    p2 = tf.keras.layers.Dropout(droupouts)(p2)
    
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    p3 = tf.keras.layers.Dropout(droupouts)(p3)
    
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
    p4 = tf.keras.layers.Dropout(droupouts)(p4)
    
    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(droupouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(droupouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(droupouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(droupouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    output = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(c9)
    model = tf.keras.Model(inputs = [inputImage], outputs = [output])
    return model
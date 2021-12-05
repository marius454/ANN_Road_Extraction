import tensorflow as tf
import matplotlib.pyplot as plt

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# def display3(display_list):
#   plt.figure(figsize=(15, 15))

#   title = ['Input Image', 'True Mask', 'Predicted Mask']

#   for display_row in display_list:
#     for i in range(len(display_row)):
#       plt.subplot(1, len(display_row), i+1)
#       plt.title(title[i])
#       plt.imshow(tf.keras.utils.array_to_img(display_row[i]))
#       plt.axis('off')
#   plt.show()
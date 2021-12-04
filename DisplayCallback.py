import Model
import tensorflow as tf
from IPython.display import clear_output

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    if (epoch + 1) % 10 == 0:
      Model.show_predictions()
      print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
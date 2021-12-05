import Model
import tensorflow as tf
from IPython.display import clear_output
import variables as var

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    if (epoch + 1) % var.EPOCHS == 0:
      print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
      Model.show_predictions()
      # Model.show_predictions(num = 3)
      

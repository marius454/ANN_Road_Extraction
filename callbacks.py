import Model
import tensorflow as tf
from IPython.display import clear_output
import variables as var

# Currently shows a random sample image at the end of training
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    # Change var.EPOCHS to a lower number to get sample images more often
    # if (epoch + 1) % var.EPOCHS == 0:
    if ((epoch + 1) % 2 == 0) or ((epoch + 1) % var.EPOCHS == 0):
      print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
      Model.show_predictions()
      

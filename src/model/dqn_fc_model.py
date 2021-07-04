import tensorflow as tf
from tensorflow import keras

class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__()
    self.hidden_layer = keras.layers.Dense(128, activation='relu')
    self.output_layer = keras.layers.Dense(num_actions)
  
  def call(self, inputs: tf.Tensor):
    x = self.hidden_layer(inputs)
    q_values = self.output_layer(x)
    return q_values
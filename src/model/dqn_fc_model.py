import tensorflow as tf
from tensorflow import keras

class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__()
    self.fc1 = keras.layers.Dense(64, activation='tanh')
    self.fc2 = keras.layers.Dense(64, activation='tanh')
    self.fc3 = keras.layers.Dense(256, activation='relu')
    self.output_layer = keras.layers.Dense(num_actions)
  
  def call(self, inputs: tf.Tensor):
    x = self.fc1(inputs)
    x = self.fc2(x)
    x = self.fc3(x)
    q_values = self.output_layer(x)
    return q_values
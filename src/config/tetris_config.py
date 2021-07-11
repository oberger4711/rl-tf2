from os import name
import tensorflow as tf
from tensorflow import keras

from .base_config import BaseConfig, CONFIGS

class TetrisConfig(BaseConfig):
  ENV_NAME = 'Tetris8x20-v0'

  def __init__(self):
    BaseConfig.__init__(self, TetrisConfig.ENV_NAME)
    self.LEARNING_RATE = 5e-4
    self.BATCH_SIZE = 32
    self.NUM_ITERATIONS_TRAINING = 300000
    self.NUM_ITERATIONS_BETWEEN_TARGET_UPDATES = 500
    self.REPLAY_MEMORY_CAPACITY = 50000
    self.EXPL_EPSILON_START = 1.0
    self.EXPL_EPSILON_END = 0.1
    self.EXPL_EPSILON_PERCENTAGE_RAMP = 0.1
    self.DISCOUNT_FACTOR = 0.95 # Gamma
    self.DOUBLE_DQN = True
    self.MONITORING_SLIDING_WINDOW_LEN = 200
    self.VIZ_TRAINING = True
    self.VIZ_TRAINING_FRAME_INTERVAL = 3
    self.VIZ_DELAY_IN_S = 0.1

  class Model(tf.keras.Model):
    def __init__(self, env):
      super().__init__()
      self.num_actions = env.action_space.n
      self.conv1 = keras.layers.Conv2D(32, 4, strides=2, activation='relu')
      self.conv2 = keras.layers.Conv2D(64, 2, strides=1, activation='relu')
      self.fc1 = keras.layers.Dense(256, activation='relu')
      self.fc2 = keras.layers.Dense(256, activation='relu')
      self.output_layer = keras.layers.Dense(self.num_actions, name='predictions')
    
    def call(self, inputs: tf.Tensor):
      x = tf.expand_dims(inputs, -1) # Add channel dim
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.fc1(tf.reshape(x, (tf.shape(inputs)[0], -1)))
      x = self.fc2(x)
      q_values = self.output_layer(x)
      return q_values

CONFIGS[TetrisConfig.ENV_NAME] = TetrisConfig
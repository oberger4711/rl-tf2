import tensorflow as tf
from tensorflow import keras

from .base_config import BaseConfig, CONFIGS

class CartPoleConfig(BaseConfig):
  ENV_NAME = 'CartPole-v1'

  def __init__(self):
    BaseConfig.__init__(self, CartPoleConfig.ENV_NAME)
    self.LEARNING_RATE = 5e-4
    self.BATCH_SIZE = 32
    self.NUM_ITERATIONS_TRAINING = 10000
    self.NUM_ITERATIONS_BETWEEN_TARGET_UPDATES = 500
    self.REPLAY_MEMORY_CAPACITY = 50000
    self.EXPL_EPSILON_START = 1.0
    self.EXPL_EPSILON_END = 0.1
    self.DISCOUNT_FACTOR = 1.0 # Gamma
    self.DOUBLE_DQN = True
    self.MONITORING_SLIDING_WINDOW_LEN = 200

    # Geometric series limes
    if self.DISCOUNT_FACTOR < 1.0: print(f'q -> {1 / (1 - self.DISCOUNT_FACTOR)}')

  class Model(tf.keras.Model):
    def __init__(self, env):
      super().__init__()
      num_actions = env.action_space.n
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

CONFIGS[CartPoleConfig.ENV_NAME] = CartPoleConfig
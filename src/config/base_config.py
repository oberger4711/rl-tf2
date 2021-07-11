CONFIGS = {}

class BaseConfig:
  def __init__(self, env_name):
    self.ENV_NAME = env_name
    # For example hyper parameters, see this implementation: https://github.com/openai/baselines/blob/master/baselines/deepq/deepq.py#L95
    # These are reasonable values:
    '''
    self.LEARNING_RATE = 5e-4
    self.BATCH_SIZE = 32
    self.NUM_ITERATIONS_TRAINING = 100000
    self.NUM_ITERATIONS_BETWEEN_TARGET_UPDATES = 500
    self.REPLAY_MEMORY_CAPACITY = 50000
    self.EXPL_EPSILON_START = 1.0
    self.EXPL_EPSILON_END = 0.10
    self.EXPL_EPSILON_PERCENTAGE_RAMP = 0.1
    self.DISCOUNT_FACTOR = 0.95 # Gamma
    self.DOUBLE_DQN = True
    self.MONITORING_SLIDING_WINDOW_LEN = 200
    '''
    self.VIZ_TRAINING = False
    self.VIZ_TRAINING_FRAME_INTERVAL = 1
    self.VIZ_DELAY_IN_S = 0
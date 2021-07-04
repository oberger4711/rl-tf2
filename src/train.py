import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Set Tensorflow log level. 0: INFO, 1: WARNING, 2: ERROR
import tensorflow as tf
import numpy as np
import gym
import tqdm

from model.dqn_fc_model import Model
from utilities.ring_buffer import RingBuffer

FLOAT_EPSILON = np.finfo(np.float32).eps.item()
ENV_NAME = 'CartPole-v0'

# DQN HYPERPARAMETERS
# For example values, see this implementation: https://github.com/openai/baselines/blob/master/baselines/deepq/deepq.py#L95
BATCH_SIZE = 32
REPLAY_MEMORY_CAPACITY = 50000
MAX_STEPS_PER_EPISODE = 500 # NOTE: Make sure there can be enough episodes in the replay memory to avoid bias
EXPL_EPSILON_INITIAL = 1.0
EXPL_EPSILON_FINAL = 0.1

envs = [gym.make(ENV_NAME)] * BATCH_SIZE
env_0 = envs[0]
envs_states = [e.reset().astype(np.float32) for e in envs]
print(f'Environment {ENV_NAME}')
print(f'  State / observation space:  {env_0.observation_space}')
print(f'  Action space:       {env_0.action_space}')

# Set seeds for more reproducible experiments
seed = 42
for i, e in enumerate(envs): e.seed(i * seed)
tf.random.set_seed(seed)
np.random.seed(seed)

q_model = Model(env_0.action_space.n)

def envs_step(actions: np.ndarray):
  transitions = [None] * BATCH_SIZE
  for i, env in enumerate(envs):
    action = actions[i]
    state_t = envs_states[i]
    raw_state_t1, raw_reward, raw_done, _ = env.step(action)
    state_t1 = raw_state_t1.astype(np.float32)
    reward = np.array(raw_reward, np.int32)
    done = np.array(raw_done, np.int32)
    transitions[i] = (state_t, actions[i], state_t1, reward, done)
    # Update tracked env state for next step
    if not raw_done:
      envs_states[i] = state_t1
    else:
      # New episode
      envs_states[i] = env.reset().astype(np.float32)
  return transitions

def decide_action_epsilon_greedy(q_values: tf.Tensor, expl_epsilon: float):
  best_q_actions = tf.argmax(q_values, axis=1, output_type=tf.int32)
  random_actions = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=q_values.shape[1], dtype=tf.int32)
  chose_random = tf.random.uniform((BATCH_SIZE,), minval=0.0, maxval=1.0, dtype=tf.float32) < expl_epsilon
  actions = tf.where(chose_random, random_actions, best_q_actions)
  return actions

def run_steps(q_model: tf.keras.Model, expl_epsilon: float):
  states_t = np.stack(envs_states)
  q_values = q_model(states_t)
  actions = decide_action_epsilon_greedy(q_values, expl_epsilon)
  transitions = envs_step(actions.numpy())
  return transitions

def init_replay_memory(q_model):
  """Initializes replay memory with data."""
  replay_memory = RingBuffer(REPLAY_MEMORY_CAPACITY)
  with tqdm.tqdm(total=REPLAY_MEMORY_CAPACITY, unit=" transitions", desc="Initializing replay memory with transitions") as t:
    while not replay_memory.full:
      transitions = run_steps(q_model, EXPL_EPSILON_INITIAL)
      replay_memory.put_all(transitions)
      t.update(len(transitions))
  return replay_memory
replay_memory = init_replay_memory(q_model)

def print_replay_memory_stats(replay_memory):
  """Print statistics over the given replay memory for debugging."""
  print("Replay memory statistics:")
  def stats_str(a):
    return f"mean = {np.mean(a):.2f},  stddev = {np.std(a):.2f},  min = {np.min(a):.2f},  max = {np.max(a):.2f}"
  initial_actions = np.array([a for (_, a, _, _, _) in replay_memory.data])
  initial_rewards = np.array([r for (_, _, _, r, _) in replay_memory.data])
  initial_done = np.array([d for (_, _, _, _, d) in replay_memory.data])
  print(f"  Actions: {stats_str(initial_actions)}")
  print(f"  Rewards: {stats_str(initial_rewards)}")
  print(f"  Done:    {stats_str(initial_done)}")
print_replay_memory_stats(replay_memory)

print("Done.")
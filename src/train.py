import os
import datetime
import time
from utilities.ring_buffer import RingBuffer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Set Tensorflow log level. 0: INFO, 1: WARNING, 2: ERROR
import tensorflow as tf
import numpy as np
import gym
import tqdm

from model.dqn_fc_model import Model
import utilities

FLOAT_EPSILON = np.finfo(np.float32).eps.item()
ENV_NAME = 'CartPole-v0'

# HYPERPARAMETERS
# For example values, see this implementation: https://github.com/openai/baselines/blob/master/baselines/deepq/deepq.py#L95
LEARNING_RATE = 5e-4 #0.001
BATCH_SIZE = 32
NUM_ITERATIONS_TRAINING = 30000
NUM_ITERATIONS_BETWEEN_TARGET_UPDATES = 500
REPLAY_MEMORY_CAPACITY = 5000 # 50000
EXPL_EPSILON_START = 1.0
EXPL_EPSILON_END = 0.1
DISCOUNT_FACTOR = 0.95 # Gamma
DOUBLE_DQN = False

# Geometric series
if ENV_NAME == 'CartPole-v0': print(f'q -> {1 / (1 - DISCOUNT_FACTOR)}')

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

# Setup logging for Tensorboard
training_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = f'logs/{ENV_NAME}/{training_time}/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

print('Initialize q model and target model.')
q_model = Model(env_0.action_space.n)
target_model = Model(env_0.action_space.n)

# Initialize models and graph by running the models once.
q_model(np.stack(envs_states))
target_model(np.stack(envs_states))

def update_target_model(q_model, target_model):
  """Copy the weights from q model to target model as described in DQN paper."""
  target_model.set_weights(q_model.get_weights())
update_target_model(q_model, target_model)

def envs_step(actions: np.ndarray):
  transitions = [None] * BATCH_SIZE
  for i, env in enumerate(envs):
    action = actions[i]
    state_t = envs_states[i]
    raw_state_t1, raw_reward, raw_done, _ = env.step(action)
    state_t1 = raw_state_t1.astype(np.float32)
    reward = np.array(raw_reward, np.float32)
    done = np.array(raw_done, np.bool)
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

def run_steps(q_model: tf.keras.Model, expl_epsilon: float=0):
  states_t = np.stack(envs_states)
  q_values = q_model(states_t)
  actions = decide_action_epsilon_greedy(q_values, expl_epsilon)
  transitions = envs_step(actions.numpy())
  return transitions, q_values.numpy()

def init_replay_memory(q_model):
  """Initializes replay memory with data."""
  replay_memory = utilities.RingBuffer(REPLAY_MEMORY_CAPACITY)
  with tqdm.tqdm(total=REPLAY_MEMORY_CAPACITY, unit=" transitions", desc="Initialize replay memory with transitions") as t:
    while not replay_memory.full:
      transitions, _ = run_steps(q_model, EXPL_EPSILON_START)
      replay_memory.put_all(transitions)
      t.update(len(transitions))
  return replay_memory
replay_memory = init_replay_memory(q_model)

def print_replay_memory_stats(replay_memory):
  """Print statistics over the given replay memory for debugging."""
  print('Replay memory statistics:')
  def stats_str(a):
    return f'mean = {np.mean(a):.2f},  stddev = {np.std(a):.2f},  min = {np.min(a):.2f},  max = {np.max(a):.2f}'
  initial_actions = np.array([a for (_, a, _, _, _) in replay_memory.data])
  initial_rewards = np.array([r for (_, _, _, r, _) in replay_memory.data])
  initial_done = np.array([d for (_, _, _, _, d) in replay_memory.data])
  print(f'  Actions: {stats_str(initial_actions)}')
  print(f'  Rewards: {stats_str(initial_rewards)}')
  print(f'  Done:    {stats_str(initial_done)}')
print_replay_memory_stats(replay_memory)

@tf.function
def q_values_of_actions(q_values, actions):
  action_mask = tf.one_hot(actions, q_values.shape[1], dtype=tf.float32)
  return tf.reduce_sum(q_values * action_mask, axis=1) # Eliminate action space dimension

# Loss and optimizer
loss_func = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) # More stable than MSE?
@tf.function
def compute_loss(mb_transitions, q_model, target_model):
  (states_t, actions, states_t1, rewards, dones) = mb_transitions
  # Q_target(s_t1, a_t1, theta_target) = 
  #   if episode ongoing:   r + gamma * max_a_t1(Q(s_t1, a_t1, theta_target))
  #   if episode done:      r
  if DOUBLE_DQN:
    # Q model is used to choose the best action in next state instead of target model
    target_qs_t1 = q_values_of_actions(target_model(states_t1), tf.argmax(q_model(states_t1), axis=1))
  else:
    target_qs_t1 = tf.reduce_max(target_model(states_t1), axis=1)
  done_masks = 1.0 - tf.cast(dones, tf.float32)
  target_qs_t = rewards + done_masks * DISCOUNT_FACTOR * target_qs_t1 # See Bellman equation
  target_qs_t = tf.stop_gradient(target_qs_t) # Do not touch target net
  # Q_actual(s_t, a_t, theta_q)
  actual_qs_t = q_values_of_actions(q_model(states_t), actions)
  return loss_func(target_qs_t, actual_qs_t)
optimizer = tf.optimizers.RMSprop(LEARNING_RATE)
#optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

# TRAINING
expl_epsilon = utilities.make_epsilon_func_ramped(EXPL_EPSILON_START, EXPL_EPSILON_END, NUM_ITERATIONS_TRAINING, 0.1)
q_values_sliding_window = utilities.RingBuffer(200)
with tqdm.trange(NUM_ITERATIONS_TRAINING, desc='Training') as t:
  for iteration in t:
    # Debugging stuff
    if iteration == 0:
      tf.summary.trace_on(graph=True, profiler=False)

    # DQN ALGORITHM:
    # Update target model every few iterations
    if (iteration + 1) % NUM_ITERATIONS_BETWEEN_TARGET_UPDATES == 0:
      update_target_model(q_model, target_model)
    # Generate new transitions with current model for replay memory
    transitions, q_values = run_steps(q_model, expl_epsilon(iteration))
    replay_memory.put_all(transitions)
    q_values_sliding_window.put_all(q_values) # For monitoring
    # Sample mini batch from the replay memory
    mb_transitions = replay_memory.sample_stacked(BATCH_SIZE)
    # Compute loss
    with tf.GradientTape() as tape:
      loss = compute_loss(mb_transitions, q_model, target_model)
    # Backpropagate error
    gradients = tape.gradient(loss, q_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_model.trainable_variables))

    # Monitoring stuff
    with train_summary_writer.as_default():
      tf.summary.scalar('loss', loss, step=iteration)
      if q_values_sliding_window.full:
        q_values = np.array(q_values_sliding_window.data)
        tf.summary.scalar('predicted q_values mean', np.mean(q_values), step=iteration)
        tf.summary.scalar('predicted q_values std dev', np.std(q_values), step=iteration)
      if iteration == 0:
          tf.summary.trace_export(name="graph", step=iteration, profiler_outdir=train_log_dir)

# SHOWCASE
print("Showcase.")
while True:
  envs[0].render()
  done = run_steps(q_model)[0][0][-1]
  if done: time.sleep(0.5) # Delay when episode over
  else: time.sleep(1.0 / 30.0)
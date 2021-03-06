import os
import datetime
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Set Tensorflow log level. 0: INFO, 1: WARNING, 2: ERROR
import tensorflow as tf
import numpy as np
import gym
import tqdm

import config
import utilities
import replaybuffer
import env.tetris

# DEBUGGING
DEBUG = False # NOTE: False is significantly faster
if DEBUG:
  tf.config.run_functions_eagerly(True) # Disable tf.function decorator
FLOAT_EPSILON = np.finfo(np.float32).eps.item()

# Instantiate config defining gym environment, model and hyperparameters
ConfigClass = config.CONFIGS['Tetris8x20-v0']
#ConfigClass = config.CONFIGS['CartPole-v1']
cfg = ConfigClass()

envs = [gym.make(cfg.ENV_NAME) for _ in range(cfg.BATCH_SIZE)]
env_0 = envs[0]
envs_states = [e.reset().astype(np.float32) for e in envs]
envs_episode_lens = [0] * cfg.BATCH_SIZE
envs_episode_total_rewards = [0] * cfg.BATCH_SIZE
print(f'Environment {cfg.ENV_NAME}:')
print(f'  State / observation space:  {env_0.observation_space}')
print(f'  Action space:               {env_0.action_space}')

# Set seeds for more reproducible experiments
seed = 42
for i, e in enumerate(envs): e.seed(i * seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Setup logging for Tensorboard
training_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = f'logs/{cfg.ENV_NAME}/{training_time}'
model_dir = f'models/{cfg.ENV_NAME}/{training_time}'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
print(f'Logging at {train_log_dir}.')

print('Initialize q model and target model.')
q_model = ConfigClass.Model(env_0)
target_model = ConfigClass.Model(env_0)

# Initialize models and graph by running the models once.
q_model(np.stack(envs_states))
target_model(np.stack(envs_states))

def update_target_model(q_model, target_model):
  """Copy the weights from q model to target model as described in DQN paper."""
  target_model.set_weights(q_model.get_weights())
update_target_model(q_model, target_model)

def envs_step(actions: np.ndarray):
  transitions = [None] * cfg.BATCH_SIZE
  done_episodes_lens, done_episodes_total_rewards = [], []
  for i, env in enumerate(envs):
    action = actions[i]
    state_t = envs_states[i]
    raw_state_t1, raw_reward, raw_done, _ = env.step(action)
    state_t1 = raw_state_t1.astype(np.float32)
    reward = np.array(raw_reward, np.float32)
    done = np.array(raw_done, np.bool)
    transitions[i] = (state_t, actions[i], state_t1, reward, done)
    # Update tracked env states for next step
    envs_episode_lens[i] += 1
    envs_episode_total_rewards[i] += reward
    if not raw_done:
      envs_states[i] = state_t1
    else:
      # New episode
      envs_states[i] = env.reset().astype(np.float32)
      done_episodes_lens.append(envs_episode_lens[i])
      done_episodes_total_rewards.append(envs_episode_total_rewards[i])
      envs_episode_lens[i] = 0
      envs_episode_total_rewards[i] = 0
  return transitions, done_episodes_lens, done_episodes_total_rewards

def decide_action_epsilon_greedy(q_values: tf.Tensor, expl_epsilon: float):
  best_q_actions = tf.argmax(q_values, axis=1, output_type=tf.int32)
  random_actions = tf.random.uniform((cfg.BATCH_SIZE,), minval=0, maxval=q_values.shape[1], dtype=tf.int32)
  chose_random = tf.random.uniform((cfg.BATCH_SIZE,), minval=0.0, maxval=1.0, dtype=tf.float32) < expl_epsilon
  actions = tf.where(chose_random, random_actions, best_q_actions)
  return actions

def run_steps(q_model: tf.keras.Model, expl_epsilon: float=0):
  states_t = np.stack(envs_states)
  q_values = q_model(states_t)
  actions = decide_action_epsilon_greedy(q_values, expl_epsilon)
  transitions, done_episodes_lens, done_episodes_total_rewards = envs_step(actions.numpy())
  return transitions, q_values.numpy(), np.array(done_episodes_lens, dtype=np.int32), np.array(done_episodes_total_rewards, dtype=np.int32)

def init_replay_memory(q_model):
  """Initializes replay memory with data."""
  replay_memory = replaybuffer.RingBuffer(cfg.REPLAY_MEMORY_CAPACITY)
  with tqdm.tqdm(total=cfg.REPLAY_MEMORY_CAPACITY, unit=" transitions", desc="Initialize replay memory with transitions") as t:
    while not replay_memory.full:
      transitions, _, _, _ = run_steps(q_model, cfg.EXPL_EPSILON_START)
      replay_memory.put_all(transitions)
      t.update(len(transitions))
  return replay_memory
replay_memory = init_replay_memory(q_model)

def print_replay_memory_stats(replay_memory):
  """Print statistics over the given replay memory for debugging."""
  print('Replay memory statistics:')
  def stats_str(a):
    return f'mean = {np.mean(a):.2f},  stddev = {np.std(a):.2f},  min = {np.min(a):.2f},  max = {np.max(a):.2f}'
  initial_states = np.array([s for (s, _, _, _, _) in replay_memory.data])
  initial_actions = np.array([a for (_, a, _, _, _) in replay_memory.data])
  initial_rewards = np.array([r for (_, _, _, r, _) in replay_memory.data])
  initial_done = np.array([d for (_, _, _, _, d) in replay_memory.data])
  print(f'  States:  {stats_str(initial_states)}')
  print(f'  Actions: {stats_str(initial_actions)}')
  print(f'  Rewards: {stats_str(initial_rewards)}')
  print(f'  Done:    {stats_str(initial_done)}')
print_replay_memory_stats(replay_memory)

@tf.function
def q_values_of_actions(q_values, actions):
  action_mask = tf.one_hot(actions, q_values.shape[1], axis=1, dtype=tf.float32)
  return tf.reduce_sum(q_values * action_mask, axis=1) # Eliminate action space dimension

# LOSS AND OPTIMIZER
loss_func = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) # More stable than MSE
@tf.function
def compute_loss(mb_transitions, q_model, target_model):
  (states_t, actions, states_t1, rewards, dones) = mb_transitions
  # Q_target(s_t1, a_t1, theta_target) = 
  #   if episode ongoing:   r + gamma * max_a_t1(Q(s_t1, a_t1, theta_target))
  #   if episode done:      r
  if cfg.DOUBLE_DQN:
    # Q model is used to choose the best action in next state instead of target model
    target_qs_t1 = q_values_of_actions(target_model(states_t1), tf.argmax(q_model(states_t1), axis=1))
  else:
    target_qs_t1 = tf.reduce_max(target_model(states_t1), axis=1)
  done_masks = 1.0 - tf.cast(dones, tf.float32)
  target_qs_t = rewards + done_masks * cfg.DISCOUNT_FACTOR * target_qs_t1 # See Bellman equation
  target_qs_t = tf.stop_gradient(target_qs_t) # Do not optimize target net
  # Q_actual(s_t, a_t, theta_q)
  actual_qs_t = q_values_of_actions(q_model(states_t), actions)
  return loss_func(target_qs_t, actual_qs_t)
#optimizer = tf.optimizers.RMSprop(cfg.LEARNING_RATE)
optimizer = tf.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)

@tf.function
def train_step(mb_transitions, q_model, target_model, optimizer):
  # Compute loss
  with tf.GradientTape() as tape:
    loss = compute_loss(mb_transitions, q_model, target_model)
  # Backpropagate error
  gradients = tape.gradient(loss, q_model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, q_model.trainable_variables))
  return loss

# TRAINING
expl_epsilon = utilities.make_epsilon_func_ramped(cfg.EXPL_EPSILON_START, cfg.EXPL_EPSILON_END, cfg.NUM_ITERATIONS_TRAINING, cfg.EXPL_EPSILON_PERCENTAGE_RAMP)
rb_episodes_lens = replaybuffer.RingBuffer(cfg.MONITORING_SLIDING_WINDOW_LEN)
rb_episodes_total_rewards = replaybuffer.RingBuffer(cfg.MONITORING_SLIDING_WINDOW_LEN)
viz_frame = 0
with tqdm.trange(cfg.NUM_ITERATIONS_TRAINING, desc='Training') as t:
  for iteration in t:
    # DQN algorithm:
    # Update target model every few iterations
    if (iteration + 1) % cfg.NUM_ITERATIONS_BETWEEN_TARGET_UPDATES == 0:
      update_target_model(q_model, target_model)

    # Generate new transitions with current model for replay memory
    epsilon = expl_epsilon(iteration)
    transitions, predicted_q_values, done_episodes_lens, done_episodes_total_rewards = run_steps(q_model, epsilon)
    replay_memory.put_all(transitions)
    if cfg.VIZ_TRAINING:
      viz_frame
      viz_frame += 1
      if viz_frame == cfg.VIZ_TRAINING_FRAME_INTERVAL:
        envs[0].render()
        viz_frame = 0
    # Monitoring stuff
    rb_episodes_lens.put_all(done_episodes_lens)
    rb_episodes_total_rewards.put_all(done_episodes_total_rewards)

    # Train one step with mini batch sampled from the replay memory
    mb_transitions = replay_memory.sample_stacked(cfg.BATCH_SIZE)
    if iteration == 0: tf.summary.trace_on(graph=True, profiler=False)
    loss = train_step(mb_transitions, q_model, target_model, optimizer)

    # Monitoring stuff
    with train_summary_writer.as_default():
      if iteration == 0:
          tf.summary.trace_export(name="graph", step=iteration, profiler_outdir=train_log_dir)
      tf.summary.scalar('loss', loss, step=iteration)
      tf.summary.scalar('epsilon', epsilon, step=iteration)
      tf.summary.scalar('predicted_q_values/mean', np.mean(predicted_q_values), step=iteration)
      tf.summary.scalar('predicted_q_values/std_dev', np.std(predicted_q_values), step=iteration)
      if rb_episodes_lens.full:
        episode_lens = np.array(rb_episodes_lens.data)
        tf.summary.scalar('episodes/length_mean', np.mean(episode_lens), step=iteration)
        #tf.summary.scalar('episodes_lengths/std_dev', np.std(episode_lens), step=iteration)
      if rb_episodes_total_rewards.full:
        episode_total_rewards = np.array(rb_episodes_total_rewards.data)
        tf.summary.scalar('episodes/total_reward_mean', np.mean(episode_total_rewards), step=iteration)
        tf.summary.scalar('episodes/total_reward_std_dev', np.std(episode_total_rewards), step=iteration)
        tf.summary.scalar('episodes/total_reward_max', np.max(episode_total_rewards), step=iteration)
        tf.summary.scalar('episodes/total_reward_min', np.min(episode_total_rewards), step=iteration)

# SAVE TRAINED MODEL
print(f'Save model at {model_dir}.')
q_model.save(model_dir)
#q_model.save_weights(model_dir)

# SHOWCASE
print("Render results.")
while True:
  envs[0].render()
  done = run_steps(q_model)[0][0][-1]
  time.sleep(cfg.VIZ_DELAY_IN_S)
  #if done: time.sleep(0.5) # Delay when episode over

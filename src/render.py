import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Set Tensorflow log level. 0: INFO, 1: WARNING, 2: ERROR
import tensorflow as tf
import numpy as np
import gym

import config
import env.tetris

def parse_args():
  parser = argparse.ArgumentParser(description='Loads a trained DQN model, lets it interact with an environment and renders the states.')
  parser.add_argument('config', type=str, help='the name of the config')
  parser.add_argument('--model-path', type=str, default='', help='the name of the saved model (default: alphabetically latest in "models/(config)/")')
  return parser.parse_args()

args = parse_args()

ConfigClass = config.CONFIGS[args.config]
#ConfigClass = config.CONFIGS['CartPole-v1']
cfg = ConfigClass()

envs = [gym.make(cfg.ENV_NAME) for _ in range(cfg.BATCH_SIZE)]
envs_states = [e.reset().astype(np.float32) for e in envs]
env_0 = envs[0]
env_0_state = envs_states[0]

# Set seeds for more reproducible experiments
seed = 42
for i, e in enumerate(envs): e.seed(i * seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Derive model path if necessary.
def find_latest_model_path(args):
  base_dir = f'models/{args.config}'
  if not os.path.isdir(base_dir):
    print(f'Error: No model-path was given and "{base_dir}" is not a directory. Cannot derive latest saved model.')
    exit(-1)
  sub_dirs = sorted([sd for sd in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{sd}')])
  if len(sub_dirs) == 0:
    print(f'Error: No model-path was given and "{base_dir}" is an empty directory. Cannot derive latest saved model.')
    exit(-1)
  return f'{base_dir}/{sub_dirs[-1]}'
model_path = args.model_path
if model_path == '': model_path = find_latest_model_path(args)
print(f'Loading model "{model_path}".')
  
q_model = tf.keras.models.load_model(model_path)
#q_model = ConfigClass.Model(env_0)
#q_model.load_weights(model_path)

def step():
  states_t = np.stack(envs_states)
  #states_t_tensor = tf.constant(states_t)
  q_values = q_model(states_t)
  actions = np.argmax(q_values, axis=1)
  envs_states[0], _, done, _ = env_0.step(actions[0])
  if done:
    envs_states[0] = env_0.reset()

# HACK: Auto render mode
auto_render = False
if 'set_auto_render_intermediate_steps' in dir(env_0.env):
  env_0.env.set_auto_render_intermediate_steps(True)
  auto_render = True
  print("Auto-rendering ON!")
else:
  print("Auto-rendering OFF!")

while True:
  step()
  if not auto_render:
    env_0.render()
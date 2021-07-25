# Reinforcement Learning Implementation with Tensorflow 2
Implementation of Deep Q Learning (DQN) algorithm with Tensorflow 2.
Environments of [OpenAI Gym](https://gym.openai.com/) are used for testing.

## Instructions
Install dependencies. You may want to use a Python virtual env:
```
pip3 install -r requirements.txt
```
Start training:
```
python3 train.py
```


## DQN: Off-policy TD Learning
Learns action values using bootstrapping.
The following loss function is minimized:

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20L%28%5Ctheta_i%29%20%3D%20%5Cmathbb%7BE%7D_%28s%2C%20a%2C%20r%2C%20s%27%29%5CBigg%5B%5CBig%28r&plus;%5Cgamma%20%5Cmax_%7Ba%27%7DQ%28s%27%2C%20a%27%3B%20%5Ctheta_%7Bi%7D%5E%7B-%7D%29%20-%20Q%28s%2C%20a%3B%20%5Ctheta_%7Bi%7D%29%5CBig%29%5E2%5CBigg%5D)

To make the algorithm more robust, the following tricks are used:

**A copy (target network) of the current model (Q network) is used for predicting the best action value of the next state**.
The weights of the target network are not touched during optimization.
Every few train steps, the learned weights of the Q network are copied to the target network.
This reduces training instability due to feedback effects when changing the weights of the Q network.

**A replay memory** is used to remember explored transitions which are then randomly sampled in a mini batch for training.
This reduces correlations between transition samples and thereby improves stability.

**Huber loss** is used instad of MSE.
The error grows quadratically for small values but for values over a given threshold the function is linear.
This reduces the impact of outliers on the optimization.
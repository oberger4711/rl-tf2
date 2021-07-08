from .tetris import Tetris

import gym
import numpy as np
import pygame

class TetrisEnv(gym.Env):
  COLORS = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
  ]
  BLACK = (0, 0, 0)
  WHITE = (255, 255, 255)
  GRAY = (128, 128, 128)

  def __init__(self, width=10, height=20):
    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(height, width))
    self.action_space = gym.spaces.Discrete(4)
    self.height = height
    self.width = width
    self.screen = None
    self.rendering = False
    self.game = Tetris(height, width)
    self.game.new_figure()
    self._get_image()

  def _get_image(self):
    img = np.array(self.game.field, dtype=np.uint8)
    if self.game.figure is not None:
      for i in range(4):
        for j in range(4):
          p = i * 4 + j
          if p in self.game.figure.image():
            img[i + self.game.figure.y, j + self.game.figure.x] = 1
    img = np.minimum(img, 1) # Clamp to [0, 1]
    #img = np.zeros((20, 10), dtype=np.uint8) # Two channels: (field, active figure)
    #img[20, 10, 0] = np.array(self.game.field, dtype=np.uint8)
    return img

  def step(self, action):
    """Run one timestep of the environment's dynamics. When end of
    episode is reached, you are responsible for calling `reset()`
    to reset this environment's state.

    Accepts an action and returns a tuple (observation, reward, done, info).

    Args:
        action (object): an action provided by the agent

    Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    """
    score_before = self.game.score
    if action == 0:
      self.game.rotate()
    elif action == 1:
      self.game.go_side(-1) # Left
    elif action == 2:
      self.game.go_side(1) # Right
    elif action == 3:
      self.game.go_space() # Drop
    else:
      raise RuntimeError("Action out of range [0, 3]")
    reward = self.game.score - score_before
    return self._get_image(), float(reward), self.game.state == 'gameover', {}

  def reset(self):
    """Resets the environment to an initial state and returns an initial
    observation.

    Note that this function should not reset the environment's random
    number generator(s); random variables in the environment's state should
    be sampled independently between multiple calls to `reset()`. In other
    words, each call of `reset()` should yield an environment suitable for
    a new episode, independent of previous episodes.

    Returns:
        observation (object): the initial observation.
    """
    self.game = Tetris(self.height, self.width)
    self.game.new_figure()
    return self._get_image()

  def render(self, mode='human'):
    """Renders the environment.

    The set of supported modes varies per environment. (And some
    environments do not support rendering at all.) By convention,
    if mode is:

    - human: render to the current display or terminal and
    return nothing. Usually for human consumption.
    - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
    representing RGB values for an x-by-y pixel image, suitable
    for turning into a video.
    - ansi: Return a string (str) or StringIO.StringIO containing a
    terminal-style text representation. The text can include newlines
    and ANSI escape sequences (e.g. for colors).

    Note:
        Make sure that your class's metadata 'render.modes' key includes
        the list of supported modes. It's recommended to call super()
        in implementations to use the functionality of this method.

    Args:
        mode (str): the mode to render with

    Example:

    class MyEnv(Env):
        metadata = {'render.modes': ['human', 'rgb_array']}

        def render(self, mode='human'):
            if mode == 'rgb_array':
                return np.array(...) # return RGB frame suitable for video
            elif mode == 'human':
                ... # pop up a window and render
            else:
                super(MyEnv, self).render(mode=mode) # just raise an exception
    """
    if not self.rendering:
      # Initialize and show window
      pygame.init()
      size = (400, 500)
      self.screen = pygame.display.set_mode(size)
      pygame.display.set_caption("Tetris")
      self.rendering = True

    self.screen.fill(TetrisEnv.WHITE)

    for i in range(self.game.height):
      for j in range(self.game.width):
        pygame.draw.rect(self.screen, TetrisEnv.GRAY, [
            self.game.x + self.game.zoom * j, self.game.y + self.game.zoom * i, self.game.zoom, self.game.zoom], 1)
        if self.game.field[i][j] > 0:
          pygame.draw.rect(self.screen, TetrisEnv.COLORS[self.game.field[i][j]],
                          [self.game.x + self.game.zoom * j + 1, self.game.y + self.game.zoom * i + 1, self.game.zoom - 2, self.game.zoom - 1])

    if self.game.figure is not None:
      for i in range(4):
        for j in range(4):
          p = i * 4 + j
          if p in self.game.figure.image():
            pygame.draw.rect(self.screen, TetrisEnv.COLORS[self.game.figure.color],
                            [self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                              self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                              self.game.zoom - 2, self.game.zoom - 2])

    font = pygame.font.SysFont('Calibri', 25, True, False)
    font1 = pygame.font.SysFont('Calibri', 65, True, False)
    text = font.render("Score: " + str(self.game.score), True, TetrisEnv.BLACK)
    text_game_over = font1.render("Game Over", True, (255, 125, 0))
    text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

    self.screen.blit(text, [0, 0])
    if self.game.state == "gameover":
      self.screen.blit(text_game_over, [20, 200])
      self.screen.blit(text_game_over1, [25, 265])

    pygame.display.flip()

  def close(self):
    """Override close in your subclass to perform any necessary cleanup.

    Environments will automatically close() themselves when
    garbage collected or when the program exits.
    """
    if self.rendering:
      pygame.quit()
      self.screen = None
      self.rendering = False

  def seed(self, seed=None):
    """Sets the seed for this env's random number generator(s).

    Note:
        Some environments use multiple pseudorandom number generators.
        We want to capture all such seeds used in order to ensure that
        there aren't accidental correlations between multiple generators.

    Returns:
        list<bigint>: Returns the list of seeds used in this env's random
        number generators. The first value in the list should be the
        "main" seed, or the value which a reproducer should pass to
        'seed'. Often, the main seed equals the provided 'seed', but
        this won't be true if seed=None, for example.
    """
    return

# Register this custom env so that it can be accessed the usual way by looking up the name
gym.envs.registration.register(
    id='TetrisFrames10x20-v0',
    entry_point='env.tetris:TetrisEnv',
    kwargs={
        'width': 10,
        'height': 20
      },
    max_episode_steps=200, # TODO: Find a good value
    reward_threshold=50    # TODO: Find a good value
)

# For debugging:
#t = TetrisEnv()
#t.render()
#obs, reward, done, _ = t.step(0)
#t.render()
#obs, reward, done, _ = t.step(3)
#t.render()
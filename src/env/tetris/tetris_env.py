import time

import gym
import numpy as np
import pygame

from .tetris import Tetris

class TetrisEnv(gym.Env):
  RENDER_INTERMEDIATE_SLEEP_IN_S = 1.0 / 15
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

  def __init__(self, width=10, height=20, render_intermediate_steps=False):
    self.height = height
    self.width = width
    self.render_intermediate_steps = render_intermediate_steps
    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(height, width))
    self.action_space = gym.spaces.Discrete(width * 4) # #translations * #rotations
    self.screen = None
    self.rendering = False
    self.game = Tetris(width, height)
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

  def __render_intermediate_step(self):
    self.render()
    time.sleep(TetrisEnv.RENDER_INTERMEDIATE_SLEEP_IN_S)

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
    if self.game.state == 'gameover':
      gym.logger.warn(
          "You are calling 'step()' even though this "
          "environment has already returned done = True. You "
          "should always call 'reset()' once you receive 'done = "
          "True' -- any further steps are undefined behavior."
      )
      return self._get_image(), 0.0, True, {}
    if action > self.action_space.n:
      gym.logger.warn(f"Action out of range [0, {self.action_space.n}]")
    score_before = self.game.score
    # First rotate then translate so that rotated pieces can reach the left and right border
    # Rotate
    rotation = action % 4
    if not self.render_intermediate_steps:
      for _ in range(rotation):
        self.game.rotate()
    else:
      for _ in range(rotation):
        self.game.rotate()
        self.__render_intermediate_step()
    # Translate
    translation = (action // 4) - (self.game.width // 2)
    direction = np.sign(translation)
    abs_translation = abs(translation)
    if not self.render_intermediate_steps:
      for _ in range(abs_translation):
        self.game.go_side(direction)
    else:
      for _ in range(abs_translation):
        self.game.go_side(direction)
        self.__render_intermediate_step()
    # Drop piece
    self.game.go_space()
    if self.render_intermediate_steps:
      self.__render_intermediate_step()
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
    self.game = Tetris(self.width, self.height)
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
  
  def set_auto_render_intermediate_steps(self, value):
    """Non-canonical function for fancier rendering."""
    self.render_intermediate_steps = value

# Register this custom env so that it can be accessed the usual way by looking up the name
gym.envs.registration.register(
    id='Tetris10x20-v0',
    entry_point='env.tetris:TetrisEnv',
    kwargs={
        'width': 10,
        'height': 20
      },
    max_episode_steps=200, # TODO: Find a good value
    reward_threshold=50    # TODO: Find a good value
)

gym.envs.registration.register(
    id='Tetris8x20-v0',
    entry_point='env.tetris:TetrisEnv',
    kwargs={
        'width': 8,
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
#obs, reward, done, _ = t.step(5)
#t.render()
#print("done")
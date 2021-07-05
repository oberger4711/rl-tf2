import random

import numpy as np

class RingBuffer:
  """Very simple ring buffer implementation for e.g. replay memory that supports adding but not removing."""
  def __init__(self, capacity):
    self.capacity = capacity
    self.data = [None] * capacity
    self.insert_cursor = 0
    self.full = False
  
  def put(self, element):
    self.data[self.insert_cursor] = element
    self.insert_cursor += 1
    if self.insert_cursor == self.capacity:
      self.insert_cursor = 0
      self.full = True

  def put_all(self, elements):
    for e in elements:
      self.put(e)

  def sample_stacked(self, n):
    if not self.full and self.insert_cursor < n:
      raise RuntimeError('Not enough elements to sample.')
    sampled_elements = random.sample(self.data, n)
    stacked_elements = tuple(np.stack([element[i] for element in sampled_elements]) for i in range(len(sampled_elements[0])))
    return stacked_elements
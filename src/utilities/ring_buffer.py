class RingBuffer:
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
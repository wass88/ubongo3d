import numpy as np
from nputils import NpUtils

class Board:
  @staticmethod
  def from_2d(board, height = 2):
    return Board.from_3d(np.array([board] * height))

  @staticmethod
  def from_3d(board):
    board = np.array(board)
    shape = np.shape(board)
    expanded = np.max(shape)
    board = NpUtils.expand_merginal(board, expanded)
    return Board(shape, expanded, board)

  def replace(self, board):
    return Board(self.shape, self.expanded, board)

  def __init__(self, shape, expanded, board):
    self.shape = shape
    self.expanded = expanded
    self.board = board

  def place(self, block, move):
    res = self.board - block.move(move).block
    if np.min(res) >= 0:
      return self.replace(res)
    else: return None

  @classmethod
  def random_board(cls, z, y, x, num, tries = 10000):
    assert y * x >= num / z
    assert num % z == 0
    ok = False
    for i in range(tries):
      res = NpUtils.zeros(y * x)
      res[np.random.choice(range(y * x), num // z, False),] = 1
      if NpUtils.connected(res.reshape([1, y, x])):
        ok = True
        break
    if ok: return cls.from_3d(np.tile(res, z).reshape([z, y, x]))
    raise RuntimeError("Too small tries.")

  def min_connected(self):
    return NpUtils.min_connected(self.board)
    
  def print(self):
    NpUtils.print(self.board)

  def count(self):
    return np.sum(self.board)

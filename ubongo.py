import numpy as np
from itertools import permutations, chain, combinations

from random import choice
from nputils import NpUtils
from blocks import Block, BlockList, BlockCount, all_blocks

from pickleutil import PickleUtil
from svg import SVG

class Board:
  @staticmethod
  def from_2d(board, height = 2):
    pass

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

class Answers:
  def __init__(self, answers = []):
    self.answers = answers

  def replace(self, answers):
    return Answers(answers)

  def extend(self, other):
    self.answers.extend(other.answers)

  def push_all(self, ans):
    return self.replace(map(lambda p : p + [ans], self.answers))

  @staticmethod
  def dump_answer(ans):
    def intrepl(i, move):
      b, m = move
      return b.move(m).block * (2 ** i) # (2 ** i) is block id
    return NpUtils.shirnk(sum(intrepl(i, move) for i, move in enumerate(ans)))

  def dump(self):
    return (Answers.dump_answer(ans) for ans in self.answers)
  
  def print(self):
    for d in self.dump():
      NpUtils.print(d)

  def has_answer(self):
    return len(self.answers) > 0

  def print_one(self):
    assert self.has_answer()
    NpUtils.print(next(self.dump()))

class Problem:
  def __init__(self, blocks, board):
    self.blocks = blocks
    self.board = board

  def pre_place_blocks(self, remain_min = 3):
    self.blocks_moves = {}
    for block in self.blocks.blocks:
      self.blocks_moves[block.name] = block.place_all_pat(self.board, remain_min)
  
  @staticmethod
  def from_block_count(block_list, board, all_block):
    res = Problem(block_list, board)
    res.blocks_moves = all_block.placeable
    return res

  def search_space(self):
    return dict((block, len(pats)) for block, pats in self.blocks_moves.items())
  
  def print_space(self):
    for block, moves in self.blocks_moves.items():
      print("\n\n")
      block.print()
      print("\n")
      for (block, move) in moves:
        block.move(move).print()

  def placeable_check(self):
    block_sum = self.blocks.count()
    board_sum = self.board.count()
    return block_sum == board_sum, (block_sum, board_sum)

  def place_blocks(self, one = True):
    ok, counts = self.placeable_check()
    if not ok:
      print("Check Count  block:%n board:%n" % counts)
      return Answer([])
    return self.place_blocks_(one, self.board, self.blocks.blocks)

  def place_blocks_(self, one, board, blocks, rem_min = 3):
    if blocks == []:
      assert np.min(board) != 0 # Confict
      assert np.max(board) != 0 # Remain space
      return Answers([[]]) # Placed
    
    block_moves = self.blocks_moves[blocks[0].name]

    res = Answers([])
    for (block, move) in block_moves:
      placed = board.place(block, move)
      if placed != None:
        #if len(blocks) > 1 and placed.min_connected() < rem_min:
        #  continue
        anss = self.place_blocks_(one, placed, blocks[1:])
        res.extend(anss.push_all((block, move)))
        if one and res.has_answer(): return res
    return res

  def print(self):
    self.board.print()
    self.blocks.print()
  
  def print_ans(self):
    self.pre_place_blocks()
    self.place_blocks().print_one()

class ProblemBoard:
  def __init__(self, board, problems):
    self.board = board
    self.problems = problems

  @staticmethod
  def make_from_board(board, block_count, block_num, one = False):
    board_count = board.count()
    block_count.calc_placeable(board)

    conds = list(block_count.all_comb_from_board(board, block_num))
    print("COND : ", len(conds))

    res = []
    for i, blocks in enumerate(conds):
      print(i,"/",len(conds),list(b.name for b in blocks.blocks))
      assert board_count == blocks.count()
      problem = Problem.from_block_count(blocks, board, block_count)
      ans = problem.place_blocks()
      if ans.has_answer():
        print("FIND")
        res.append(problem)
        if one: return ProblemBoard(board, res)

    print("FIND : ", len(res))
    return ProblemBoard(board, res)

  def len(self):
    return len(self.problems)
  
  def print_one(self):
    self.problems[0].print_ans()

class ProblemSet:
  def __init__(self, probboards):
    self.probboards = probboards

  @staticmethod
  def make(block_count, player):
    res = []
    blockset = block_count.split_blocks(player)
    for blocks in blockset:
      while True:
        cond_c = [20, 22, 24]
        board = Board.random_board(2, 5, 4, choice(cond_c))
        board.print()
        problemb = ProblemBoard.make_from_board(board, blocks, 5)
        if len(problemb.problems) >= 3:
          problemb.print_one()
          res.append(problemb)
          break
    return ProblemSet(res)
  
def test_solver():
  alls = all_blocks()

  board = Board.random_board(2, 2, 4, 16)
  board.print()
  blocks = alls.get_blocklist().sublist(["BV", "BV", "BP", "BP"])
  problem = Problem(blocks, board)
  problem.print_ans()

  PickleUtil.write("data/sample5", problem)
  prob = PickleUtil.read("data/sample5")
  prob.print_ans()

  probset = ProblemSet.make(alls, 5)
  PickleUtil.write("data/5p", probset)
  PickleUtil.read("data/5p")

def test_svg():
  alls = all_blocks()
  board = Board.random_board(2, 2, 4, 16)
  board.print()
  blocks = alls.get_blocklist().sublist(["BV", "BV", "BP", "BP"])
  problem = Problem(blocks, board)
  problem.print_ans()

  svg = SVG.svg()
  g = SVG.gtrans(3, 3)
  g.append(SVG.board(board))
  svg.append(g)

  g = SVG.gtrans(6, 3)
  blocks = alls.get_blocklist().sublist(["BV", "YV", "RV"])
  g.append(SVG.blocklist(blocks))
  svg.append(g)

  svg.save("data/test.svg")
  print(svg.concat())

if __name__ == "__main__":
  #test_solver()
  test_svg()
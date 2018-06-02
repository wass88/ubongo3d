import numpy as np
from itertools import permutations, chain, combinations

from random import choice, shuffle
from nputils import NpUtils
from blocks import Block, BlockList, BlockCount, all_blocks

from pickleutil import PickleUtil
from svg import SVG

from multiprocessing import Pool, Process

import datetime

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
      return Answers([])
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

def solve(data):
  (i, (blocks, board, block_count)) = data
  print(i,":",list(b.name for b in blocks.blocks))
  problem = Problem.from_block_count(blocks, board, block_count)
  ans = problem.place_blocks()
  if ans.has_answer():
    print("FIND")
    return problem
  return None

class ProblemBoard:
  def __init__(self, board, problems):
    self.board = board
    self.problems = problems

  @staticmethod
  def make_from_board(board, block_count, block_num, count = 10000000, 
                        rand = False, para = False, itr = False):
    board_count = board.count()
    block_count = block_count.clone()
    block_count.calc_placeable(board)

    conds = list(block_count.all_comb_from_board(board, block_num))
    if rand: shuffle(conds)
    print("COND : ", len(conds))

    params = enumerate(map(lambda blocks: [blocks, board, block_count], conds))
    if para:
      pool = Pool(8)
      res = pool.map(solve, params)
      res = list(filter(lambda x : x != None, res))
    else:
      res = []
      for ib in params:
        problem = solve(ib)
        if itr:
          yield problem
          if problem:
            res.append(problem)
        if len(res) >= count: return ProblemBoard(board, res)

    print("FIND : ", len(res))
    if itr:
      print("END")
    else:
      return ProblemBoard(board, res)

  def len(self):
    return len(self.problems)
  
  def append(self, problem):
    assert self.board == problem.board
    self.problems.append(problem)

  def extend(self, problemb):
    self.problems.extend(problemb.problems)
  
  def remove_index(self, index):
    del self.problems[index]
  
  def print_one(self):
    self.problems[0].print_ans()

class ProblemSet:
  def __init__(self, probboards):
    self.probboards = probboards

  def replace(self, probboards):
    return ProblemSet(probboards)

  @staticmethod
  def make(setting):
    block_count = setting.block_count
    player = setting.player
    puzzles = setting.puzzles
    block_num = setting.block_num
    cond_size = setting.board_nums

    pre_search = 20
    pre_req = 2

    boards = []
    for i in range(player):
      print("PreSearch: ", i)
      ok = False
      while not ok:
        c = 0
        board = Board.random_board(2, setting.board_width, setting.board_height, choice(cond_size))
        probb = ProblemBoard.make_from_board(board, block_count, block_num, rand=True, itr=True)
        for _ in range(pre_search):
          p = next(probb)
          if p:
            c += 1
            ok = c >= pre_req
          if c >= pre_req:
            break
      boards.append(board)
      
    res = [ProblemBoard(b, []) for b in boards]
    ids = list(range(player))
    for i in range(puzzles):
      ok = False
      while not ok: 
        counts = block_count.clone()
        shuffle(ids)
        ok = True
        r = [None for _ in range(player)]
        for j in ids:
          print("Puzzle:", i, " FOR: ", j)
          probs = ProblemBoard.make_from_board(boards[j], counts, block_num, rand=True, itr=True)
          for prob in probs:
            if prob: break
          if not prob:
            ok = False
            print("Retry")
            break
          assert counts.is_include_list(prob.blocks)
          r[j] = prob
          counts = counts.remove_list(prob.blocks)
      print("Puzzle:", i, "Satisfied")
      for j in ids:
        res[j].append(r[j])

    return ProblemSet(res)
    
  def set_name(self, name):
    self.name = name

class GameSetting:
  def __init__(self, block_count, games, player, puzzles, block_num, board_nums, board_width, board_height):
    self.block_count = block_count
    self.games = games
    self.player = player
    self.puzzles = puzzles
    self.block_num = block_num
    self.board_nums = board_nums
    self.board_width = board_width
    self.board_height = board_height

class Game:
  def __init__(self, probsets):
    self.probsets = probsets

  @staticmethod
  def make(setting):
    res = []
    for i in range(setting.games):
      print("Making: ", i)
      s = ProblemSet.make(setting)
      PickleUtil.write("data/make2_"+str(i), s)
      res.append(s)
    return Game(res)
  
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

def make_game():
  setting = GameSetting(
    block_count = all_blocks(),
    games = 10,
    player = 4,
    puzzles = 4,
    block_num = 5,
    board_nums = [20, 22, 24],
    board_width = 5,
    board_height = 4
  )

  game = PickleUtil.read("data/game4player5block")
  s = ProblemSet.make(setting)
  game.probsets[1] = s
  PickleUtil.write("data/game4player5block.2", game)
  
  #game = Game.make(setting)
  #PickleUtil.write("data/game4player5block", game)

def solving():
  alls = all_blocks()
  board = Board.from_2d((
    [[1, 0, 1],
     [1, 1, 1],
     [1, 1, 0],
     [1, 1, 1],
     [0, 0, 1]
    ]))
  blocks = alls.get_blocklist().sublist(["YV", "RZ", "RZ", "YT", "RL"])
  board.print()
  problem = Problem(blocks,board)

  html = SVG.html()
  game = PickleUtil.read("data/game4player5block")
  for probs in game.probsets:
    html.append(SVG.problemset(probs))
    for probb in probs.probboards:
      for prob in probb.problems:
        prob.print_ans()
  html.save("data/game.html")
    
if __name__ == "__main__":
  #test_solver()
  #make2()
 make_game()
 # test_svg()
 # solving()
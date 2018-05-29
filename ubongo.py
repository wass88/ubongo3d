import numpy as np
import pickle
from itertools import permutations, chain, combinations

from nputils import NpUtils
from blocks import Block, BlockList, BlockCount, all_blocks

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

  def pre_place_blocks(self):
    self.blocks_moves = {}
    for block in self.blocks.blocks:
      self.blocks_moves[block.name] = block.place_all_pat(self.board)
  
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

  def place_blocks_(self, one, board, blocks):
    if blocks == []:
      assert np.min(board) != 0 # Confict
      assert np.max(board) != 0 # Remain space
      return Answers([[]]) # Placed
    
    block_moves = self.blocks_moves[blocks[0].name]

    res = Answers([])
    for (block, move) in block_moves:
      placed = board.place(block, move)
      if placed != None:
        anss = self.place_blocks_(one, placed, blocks[1:])
        res.extend(anss.push_all((block, move)))
        if one and res.has_answer(): return res
    return res

  @staticmethod
  def make_from_board(board, block_count, block_num):
    board_count = board.count()
    block_count.calc_placeable(board)

    print("ST e")
    for blocks in block_count.all_comb_from_board(board, block_num):
      print(list(b.name for b in blocks.blocks))
      assert board_count == blocks.count()
      problem = Problem.from_block_count(blocks, board, block_count)
      ans = problem.place_blocks()
      if ans.has_answer():
        return problem
    print("END")
    return None
  
  def print(self):
    self.board.print()
    self.blocks.print()

class ProblemSet:
  def __init__(self, problems):
    self.problems = problems

def full_svg(svg, width = 100, height = 100):
  return ('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%dpx" height="%dpx">' %
    (width, height) +
    svg +
  '</svg>')

def svg_board(board, size = 10, width = 2):
  res = []
  for by, y in enumerate(board):
    for bx, x in enumerate(by):
      if bx != 0:
        res += ('<rect x="%d" y="%d" width="%d" height="%d"'
          'fill="none" stroke="black" stroke-width="%d" />' %
          (y * size, x * size, size, size, 2)
        )
  return res

def read_pickle(f):
  with open(f, mode="rb") as f:
   return pickle.load(f)

def write_pickle(res, f):
  with open(f, mode="wb") as f:
    pickle.dump(res, f)

def print_problems(f):
  for (r_map, result) in f:
    if len(result) >= 0:
      print(len(result))
      print([[r[2] for r in br[0]] for br in result])
      print_block(r_map)

def useable_parts(exits_parts, max_parts):
  return all((exits_parts.count(b) <= max_parts[b] for b in max_parts))

def ok_problem(problem, useds, max_parts):
  prob = []
  for results in problem[1]:
    prob += [[res[2] for res in results[0]]]
  print(prob)
  print(len(prob))
  for used in useds:
    ok = True
    for pro in prob:
      ok &= useable_parts(pro + used, max_parts)
    if not ok : return False
  return True

def calc_max_parts(blocks):
  res = {}
  for (_, n, b) in blocks:
    res[b] = n
  return res

def test_solver():
  alls = all_blocks()

  board = Board.random_board(2, 2, 4, 16)
  board.print()
  blocks = alls.get_blocklist().sublist(["BV", "BV", "BP", "BP"])
  problem = Problem(blocks, board)
  problem.pre_place_blocks()
  print(problem.search_space())
  ans = problem.place_blocks()
  ans.print_one()

  board = Board.random_board(2, 4, 4, 16)
  board.print()
  problem = Problem.make_from_board(board, alls, 4)
  problem.print()
  problem.pre_place_blocks()
  problem.place_blocks().print_one()

if __name__ == '__main__':
  test_solver()

# problems = [[
#   ["GLBPBVRO", "ROGJYTRZ", "GSYVRORV", "GLRVROGJ", "YVGJYTRO"],
#   ["GJYVBVGL", "BVGTBLGJ", "BVRLYVRZ", "GJRZYTBV", "BLBVRVGJ"],
#   ["YVBPBLRZ", "ROBSGTYV", "GTYLYVGL", "BVGSROYT", "GSRVRLYV"],
#   ["YVGJROBL", "BLRORVGJ", "ROBVBLGL", "RVGJROGL", "RORLYVGJ"]
# ]]
# problems = [[[[p[i:i+2] for i in range(0, 8, 2)] for p in ps] for ps in prs] for prs in problems]
# problems = [pr for prs in problems for pr in np.transpose(prs, (1, 0, 2))]
# problems = [[p for pr in prs for p in pr] for prs in problems]
# 
# all_blocks = [(np.array(b[0]), b[1], b[2]) for b in all_blocks]
# def make_puzzles(block_num, sizes, num):
#   r_maps = []
#   for i in range(num):
#     r_maps += [random_map(2, 5, 3, np.random.choice(sizes))]
# 
#   results = []
#   for i, r_map in enumerate(r_maps):
#     print("Make ", i)
#     result = make_problem(all_blocks, r_map, block_num, 100)
#     if len(result) >= 1:
#       select, ans = result[0]
#       for a in viewer(ans):
#         print_block(a)
#       for s in select:
#         print(s[2])
#         print_block(s[0])
#       print_block(r_map)
#       print("Find Answers: ", len(result))
# 
#       results += [(r_map, result)]
#   return results
# 
# # res = make_puzzles(5, [22, 24], 10)
# # write_pickle(res, "5block.pickle")
# 
# # res :: [
# #   ( r_map
# #   , results
# #     [
# #       ( select [
# #           ( blocks, count, name )
# #         ]
# #       , ans [
# #           ( ans, blocks )
# #         ]
# #       )
# #     ]
# #   )
# # ]
# 
# res = read_pickle("5block.pickle")
# print_problems(res)
# 
# max_parts = calc_max_parts(all_blocks)
# print(res[0][1])
# for r in res:
#   print(ok_problem(r, problems, max_parts))
# 
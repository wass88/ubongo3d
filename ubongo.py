import numpy as np
import pickle
from itertools import permutations, chain, combinations

def uniq_by(f, ns):
  mapped = iter(sorted(map(lambda n : (f(n), n), ns)))
  hd = next(mapped, None)
  if hd == None: return hd[1]
  prev = hd[0]
  res = [hd[1]]
  for fn, n in mapped:
    if prev != fn: res.append(n)
    prev = fn
  return res
assert uniq_by(lambda x : x % 2, [3,4,1,2]) == [2, 1]

class NpUtils:
  X = 2
  Y = 1
  Z = 0

  @staticmethod
  def zeros(shape):
    return np.zeros(shape, dtype=np.int)

  @classmethod
  def rotX(cls, block):
    return np.flip(np.swapaxes(block, cls.Y, cls.Z), cls.Z)

  @classmethod
  def rotY(cls, block):
    return np.flip(np.swapaxes(block, cls.Z, cls.X), cls.X)

  @classmethod
  def rotZ(cls, block):
    return np.flip(np.swapaxes(block, cls.X, cls.Y), cls.Y)

  @classmethod
  def rotXYZ(cls, block, x, y, z):
    res = block
    for i in range(x): res = cls.rotX(res)
    for i in range(y): res = cls.rotY(res)
    for i in range(z): res = cls.rotZ(res)
    return res
    
  @staticmethod
  def lead_zero(a):
    res = 0
    for c in a:
      if c == 0: res += 1
      else: break
    return res
    
  @classmethod
  def shirnk(cls, block):
    zz = np.amax(np.amax(block, 2), 1)
    zy = np.amax(np.amax(block, 0), 1)
    zx = np.amax(np.amax(block, 0), 0)
    sz = cls.lead_zero(zz)
    sy = cls.lead_zero(zy)
    sx = cls.lead_zero(zx)
    tz = cls.lead_zero(reversed(zz))
    ty = cls.lead_zero(reversed(zy))
    tx = cls.lead_zero(reversed(zx))
    return block[sz:-tz, sy:-ty, sx:-tx]
  
  rotations = [
    (x, y, z) for x in range(4) for y in range(4) for z in range(4)
  ]

  @classmethod
  def expand_merginal(cls, board, num):
    return cls.expand(board, 0, num, 0, num, 0, num)

  normal_rots = [(0, y, z) for y in range(4) for z in range(4)] + \
                [(1, 2 * y, z) for y in range(2) for z in range(4)]

  @classmethod
  def expand(cls, block, lz, rz, ly, ry, lx, rx):
    shape = block.shape
    block = np.concatenate((cls.zeros([shape[cls.Z], shape[cls.Y], lx]),
                  block,
                  cls.zeros([shape[cls.Z], shape[cls.Y], rx])), cls.X)
    shape = block.shape
    block = np.concatenate((cls.zeros([shape[cls.Z], ly, shape[cls.X]]),
                  block,
                  cls.zeros([shape[cls.Z], ry, shape[cls.X]])), cls.Y)
    shape = block.shape
    block = np.concatenate((cls.zeros([lz, shape[cls.Y], shape[cls.X]]),
                  block,
                  cls.zeros([rz, shape[cls.Y], shape[cls.X]])), cls.Z)
    return block
  
  @classmethod
  def expand_fit(cls, inner, outer):
    boz, boy, box = outer.shape
    blz, bly, blx = inner.shape
    return cls.expand(inner, 0, boz - blz, 0, boy - bly, 0, box - blx)
  
  @classmethod
  def connected(cls, block):
    block = np.copy(block)
    for z in range(len(block)):
      for y in range(len(block[0])):
        for x in range(len(block[0][0])):
          if block[z][y][x] != 0:
            cls.remove_con(block, z, y, x)
            return np.max(block) == 0
    return True

  D6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
  @classmethod
  def remove_con(cls, block, z, y, x):
    if 0 <= z < len(block) and 0 <= y < len(block[0]) and 0 <= x < len(block[0][0]):
      if block[z][y][x] != 0:
        block[z][y][x] = 0
        for (dz, dy, dx) in cls.D6:
          remove_con(block, z + dz, y + dy, x + dx)

  normal_rots = [(0, y, z) for y in range(4) for z in range(4)] + \
                [(1, 2 * y, z) for y in range(2) for z in range(4)]
  @staticmethod
  def print(block):
    lines = [""] * len(block[0])
    for pz in block:
      for i, py in enumerate(pz):
        lines[i] += "|" + "".join(format(px, "x") if px != 0 else "." for px in py)
    for line in lines:
      print(line)
    print()

class Block:
  def __init__(self, block, name = ""):
    self.name = name
    self.block = block
  
  def replace(self, block):
    return Block(block, self.name)
  
  def print(self):
    NpUtils.print(self.block)

  def flat(self):
    shape = list(self.block.shape)
    return shape + list(np.reshape(self.block, shape[NpUtils.X] * shape[NpUtils.Y] * shape[NpUtils.Z]))
  
  @staticmethod
  def from_flat(flat_block, name = ""):
    return Block(np.array(flat_block[3:]).reshape(flat_block[:3]), name)

  def move(self, move):
    z, y, x = move
    return self.replace(np.roll(np.roll(np.roll(self.block, x, NpUtils.X), y, NpUtils.Y), z, NpUtils.Z))

  def all_rots(self):
    max_w = max(self.block.shape)
    max_b = NpUtils.zeros([max_w, max_w, max_w])
    e_block = self.replace(NpUtils.expand_fit(self.block, max_b))
    rot_blocks = map(lambda r : (r, e_block.rotate(r)), NpUtils.normal_rots)
    res = uniq_by(lambda b : b[1].flat(), rot_blocks)
    return res

  def alimnent(self):
    sz = NpUtils.lead_zero(np.amax(np.amax(self.block, 2), 1))
    sy = NpUtils.lead_zero(np.amax(np.amax(self.block, 0), 1))
    sx = NpUtils.lead_zero(np.amax(np.amax(self.block, 0), 0))
    return self.move((-sz, -sy, -sx))

  def rotate(self, r):
    return self.replace(NpUtils.rotXYZ(self.block, *r)).alimnent()

  def place_all_pat(self, board):
    rot_blocks = map(lambda b : b[1], self.all_rots())
    e_blocks = map(lambda b : b.replace(NpUtils.expand_fit(b.block, board.board)), rot_blocks)
    res = {}
    width = board.expanded
    for block in e_blocks:
      for x in range(width): 
        for y in range(width):
          for z in range(width):
            move = (z, y, x)
            p = board.place(block, move)
            if p != None: res[(block, move)] = p
    return res # [roted_block, rot)] = board

class BlockList:
  def __init__(self, blocks):
    self.blocks = blocks

  @staticmethod
  def all_pattern(ls, counts):
    res = []
    prod = np.prod(ls)
    for com in chain.from_iterable(
      map((lambda x: combinations(range(prod),x)), counts)):
      raw = NpUtils.zeros([prod])
      raw[com,] = 1
      raw = raw.reshape(ls)
      if NpUtils.connected(raw):
        res.extend(Block(raw).all_rot())

    res = uniq_by(lambda x : x.flat(), res)
    return res

  @staticmethod
  def random_blocks(num):
    return [self.blocks[i] for i in np.random.choice(range(len(all_blocks)), num, False)]

class Board:
  @staticmethod
  def from_2d(board, height = 2):
    pass

  @staticmethod
  def from_3d(board):
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

  @staticmethod
  def random_map(z, y, x, num):
    assert y * x >= num / z
    while True:
      res = zeros(y * x)
      res[np.random.choice(range(y * x), num // z, False),] = 1
      if NpUtils.connected(res.reshape([1, y, x])):
        break
    return np.tile(res, z).reshape([z, y, x])
  
  def print(self):
    NpUtils.print(self.board)

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

class Problem:
  def __init__(self, blocks, board):
    self.blocks = blocks
    self.board = board

  def pre_place_blocks(self):
    self.blocks_moves = {}
    for block in self.blocks.blocks:
      self.blocks_moves[block] = block.place_all_pat(self.board)

  def search_space(self):
    return dict((block, len(pats)) for block, pats in self.blocks_moves.items())
  
  def print_space(self):
    for block, moves in self.blocks_moves.items():
      print("\n\n")
      block.print()
      print("\n")
      for (block, move) in moves:
        block.move(move).print()

  def place_blocks(self):
    return self.place_blocks_(self.board, list(self.blocks_moves.items()))

  def place_blocks_(self, board, blocks_moves):
    if blocks_moves == []:
      assert np.min(board) != 0 # Confict
      assert np.max(board) != 0 # Remain space
      return Answers([[]]) # Placed

    orig_block, block_moves = blocks_moves[0]

    res = Answers([])
    for (block, move) in block_moves:
      placed = board.place(block, move)
      if placed != None:
        anss = self.place_blocks_(placed, blocks_moves[1:])
        res.extend(anss.push_all((block, move)))
    return res

  @staticmethod
  def make_problem(blocks, board, num, max_counts):
    board_sum = np.sum(board)
    res = []
    e_board, all_moves = pre_place(board, blocks, 5)
    res_counts = 0
    for select in combinations(all_moves, num):
      select_moves = [s[0] for s in select]
      counts = np.sum([s[1][0] for s in select])
      if counts == board_sum:
        result = place_all(e_board, select_moves)
        if len(result) != 0:
          res += [([s[1] for s in select], result)]
          print("FIND")
          res_counts += 1
          if res_counts >= max_counts:
            return res
    return res

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

bv = Block(np.array([
  [[1, 1, 0],
   [1, 0, 0]],
  [[0, 0, 0],
   [0, 0, 0]],
]),"V")
bl = Block(np.array([
  [[1, 1, 1],
   [1, 0, 0]],
  [[0, 0, 0],
   [0, 0, 0]],
]),"L")
bu = Block(np.array([
  [[1, 1, 1],
   [1, 0, 1]],
  [[0, 0, 0],
   [0, 0, 0]],
]),"U")

sampleblock = BlockList([bv, bl, bu])
sampleboard = Board.from_3d(np.array([
  [[1, 1, 1],
   [1, 1, 1]],
  [[1, 1, 1],
   [1, 1, 1]]
]))

sampleproblem = Problem(sampleblock, sampleboard)
sampleproblem.pre_place_blocks()
# print(sampleproblem.search_space())
# sampleproblem.print_space()
# sampleboard.print()
ans = sampleproblem.place_blocks()
ans.print()

# for view in p_view:
#   print_block(view)
# 
# all_parts = all_pattern([2,3,2], [5])
# for parts in all_parts:
#   print_block(parts)
# print(len(all_parts))
# 
# all_blocks = [
# (
# [[[1, 1, 1],
#   [0, 1, 0]],
#  [[1, 0, 0],
#   [0, 0, 0]]]
#   , 2, "YT"
# ),(
# [[[1, 1, 1],
#   [1, 0, 0]],
#  [[0, 0, 1],
#   [0, 0, 0]]]
#   , 2, "YL"
# ),(
# [[[1, 1, 0],
#   [0, 1, 0]],
#  [[0, 0, 0],
#   [0, 1, 0]]]
#   , 3, "YV"
# ),(
# [[[1, 1, 1],
#   [1, 0, 1]],
#  [[0, 0, 0],
#   [0, 0, 0]]]
#   , 2, "YU"
# ),(
# [[[1, 1, 1],
#   [0, 0, 1]],
#  [[0, 0, 0],
#   [0, 0, 1]]]
#   , 2, "BL"
# ),(
# [[[0, 1, 1],
#   [1, 1, 0]],
#  [[0, 0, 0],
#   [1, 0, 0]]]
#   , 2, "BS"
# ),(
# [[[1, 1, 1],
#   [0, 1, 1]],
#  [[0, 0, 0],
#   [0, 0, 0]]]
#   , 3, "BP"
# ),(
# [[[1, 1, 0],
#   [1, 0, 0]],
#  [[0, 0, 0],
#   [0, 0, 0]]]
#   , 4, "BV"
# ),(
# [[[1, 1, 0],
#   [1, 1, 0]],
#  [[1, 0, 0],
#   [0, 0, 0]]]
#   , 3, "RO"
# ),(
# [[[1, 1, 0],
#   [1, 0, 0]],
#  [[0, 0, 0],
#   [1, 0, 0]]]
#   , 3, "RV"
# ),(
# [[[1, 1, 1],
#   [0, 0, 1]],
#  [[1, 0, 0],
#   [0, 0, 0]]]
#   , 2, "RL"
# ),(
# [[[0, 1, 1],
#   [1, 1, 0]],
#  [[0, 0, 0],
#   [0, 0, 0]]]
#   , 2, "RZ"
# ),(
# [[[1, 1, 0],
#   [0, 1, 1]],
#  [[1, 0, 0],
#   [0, 0, 0]]]
#   , 2, "GS"
# ),(
# [[[1, 1, 1],
#   [1, 0, 0]],
#  [[0, 0, 0],
#   [1, 0, 0]]]
#   , 2, "GL"
# ),(
# [[[1, 1, 1],
#   [0, 1, 0]],
#  [[0, 0, 0],
#   [0, 0, 0]]]
#   , 2, "GT"
# ),(
# [[[1, 1, 1],
#   [0, 0, 1]],
#  [[0, 0, 0],
#   [0, 0, 0]]]
#   , 4, "GJ"
# )]
# 
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
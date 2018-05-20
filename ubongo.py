import numpy as np
import pickle
from itertools import permutations, chain, combinations

piece = np.array([
  [[1,1,1,1],
   [1,0,0,0],
   [1,0,0,0]],
  [[1,0,0,0],
   [0,0,0,0],
   [0,0,0,0]]
])

X = 2
Y = 1
Z = 0

def zeros(s):
  return np.zeros(s, dtype=np.int)
def print_block(block):
  for pz in block:
    for py in pz:
      for px in py:
        if px != 0: print(px, end="")
        else: print(".", end="")
      print()
    print("-"*len(py))
  print("="*len(py))

def rotX(block):
  return np.flip(np.swapaxes(block, Y, Z), Z)

def rotY(block):
  return np.flip(np.swapaxes(block, Z, X), X)

def rotZ(block):
  return np.flip(np.swapaxes(block, X, Y), Y)

def rotXYZ(block, x, y, z):
  res = block
  for i in range(x): res = rotX(res)
  for i in range(y): res = rotY(res)
  for i in range(z): res = rotZ(res)
  return res

rots = [
  (x, y, z) for x in range(4) for y in range(4) for z in range(4)
]

def flatten(block):
  shape = list(block.shape)
  return shape + list(np.reshape(block, shape[X] * shape[Y] * shape[Z]))

def rebuild(flat_block):
  return np.array(flat_block[3:]).reshape(flat_block[:3])

def move(block, m):
  z, y, x = m
  return np.roll(np.roll(np.roll(block, x, X), y, Y), z, Z)

def countHeadZero(a):
  res = 0
  for c in a:
    if c == 0: res += 1
    else: break
  return res

def alimnent(block):
  sz = countHeadZero(np.amax(np.amax(block, 2), 1))
  sy = countHeadZero(np.amax(np.amax(block, 0), 1))
  sx = countHeadZero(np.amax(np.amax(block, 0), 0))
  return move(block, (-sz, -sy, -sx))

def shirnk(block):
  zz = np.amax(np.amax(block, 2), 1)
  zy = np.amax(np.amax(block, 0), 1)
  zx = np.amax(np.amax(block, 0), 0)
  sz = countHeadZero(zz)
  sy = countHeadZero(zy)
  sx = countHeadZero(zx)
  tz = countHeadZero(reversed(zz))
  ty = countHeadZero(reversed(zy))
  tx = countHeadZero(reversed(zx))
  return block[sz:-tz, sy:-ty, sx:-tx]

def available_rot(block, rots):
  rot = []
  max_w = max(block.shape)
  max_b = zeros([max_w, max_w, max_w])
  block = expand_fit(max_b, block)
  for r in rots:
    rot += [(flatten(alimnent(rotXYZ(block, *r))), r)]

  rot.sort()
  res = [(rebuild(rot[0][0]), rot[0][1])]
  for i in range(len(rot) - 1):
    if rot[i+1][0] != rot[i][0]:
      res += [(rebuild(rot[i+1][0]), rot[i+1][1])]

  return res

def calc_normal_rots(block, rots):
  res = []
  for a, b in available_rot(piece, rots):
    res += [b]
  res.sort()
  return res

def expand_merginal(board, num):
  return expand(board, 0, num, 0, num, 0, num)

def expand_fit(board, block):
  boz, boy, box = board.shape
  blz, bly, blx = block.shape
  return expand(block, 0, boz - blz, 0, boy - bly, 0, box - blx)

normal_rots = [(0, y, z) for y in range(4) for z in range(4)] + \
              [(1, 2 * y, z) for y in range(2) for z in range(4)]

def expand(block, lz, rz, ly, ry, lx, rx):
  shape = block.shape
  block = np.concatenate((zeros([shape[Z], shape[Y], lx]),
                 block,
                 zeros([shape[Z], shape[Y], rx])), X)
  shape = block.shape
  block = np.concatenate((zeros([shape[Z], ly, shape[X]]),
                block,
                zeros([shape[Z], ry, shape[X]])), Y)
  shape = block.shape
  block = np.concatenate((zeros([lz, shape[Y], shape[X]]),
                block,
                zeros([rz, shape[Y], shape[X]])), Z)
  return block

assert normal_rots == calc_normal_rots(piece, rots)

def pre_place(board, blocks, width):
  e_board = expand_merginal(board, width - 1)
  all_moves = [(place_move(e_board, block[0], width), block) for block in blocks]
  return (e_board, all_moves)

def place_all(e_board, all_moves):
  print("START", [len(m) for m in all_moves])
  return place_all_(e_board, all_moves)

def place_all_(board, all_moves):
  if len(all_moves) == 0:
    assert np.min(board) == 0
    if np.max(board) == 0:
      return [[]]
    else:
      return []
  moves, *remain = all_moves
  res = []
  for (_, block, m)  in moves:
    placed = place(board, block, m)
    if placed is not None:
      res += [[(block, m)] + p for p in place_all_(placed, remain)]
  return res

def place_move(e_board, block, width):
  rot_blocks = [b[0] for b in available_rot(block, normal_rots)]
  #for b in rot_blocks: print_block(b)
  e_blocks = [expand_fit(e_board, b) for b in rot_blocks]
  all_moves = [(place(e_board, b, (z, y, x)), b, (z,y,x)) for b in e_blocks
    for x in range(width) for y in range(width) for z in range(width)]
  ok_moves = [m for m in all_moves if m[0] is not None]
  return ok_moves

def place(board, block, m):
  res = board - move(block, m)
  if np.min(res) >= 0:
    return res
  else: return None

def connected(block):
  block = np.copy(block)
  for z in range(len(block)):
    for y in range(len(block[0])):
      for x in range(len(block[0][0])):
        if block[z][y][x] != 0:
          remove_con(block, z, y, x)
          return np.max(block) == 0

D6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
def remove_con(block, z, y, x):
  if 0 <= z < len(block) and 0 <= y < len(block[0]) and 0 <= x < len(block[0][0]):
    if block[z][y][x] != 0:
      block[z][y][x] = 0
      for (dz, dy, dx) in D6:
        remove_con(block, z + dz, y + dy, x + dx)

def all_pattern(ls, counts):
  res = []
  prod = np.prod(ls)
  for com in chain.from_iterable(
    map((lambda x: combinations(range(prod),x)), counts)):
    raw = zeros([prod])
    raw[com,] = 1
    raw = raw.reshape(ls)
    if connected(raw):
      res += [flatten(available_rot(raw, normal_rots)[0][0])]

  res.sort()
  rr = [rebuild(res[0])]
  for i in range(len(res)-1):
    if res[i-1] != res[i]:
      rr += [rebuild(res[i])]
  return rr

def viewer(p_all):
  return [shirnk(sum([move(b, m) * (2 ** i) for i, (b, m) in enumerate(pa)])) for pa in p_all]

def random_blocks(all_blocks, num):
  return [all_blocks[i] for i in np.random.choice(range(len(all_blocks)), num, False)]

def random_map(z, y, x, num):
  assert y * x >= num / z
  while True:
    res = zeros(y * x)
    res[np.random.choice(range(y * x), num // z, False),] = 1
    if connected(res.reshape([1, y, x])):
      break
  return np.tile(res, z).reshape([z, y, x])

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

blocks = [
(np.array([
  [[1, 1, 0],
   [1, 0, 0]],
  [[0, 0, 0],
   [0, 0, 0]],
]),"V"),
(np.array([
  [[1, 1, 1],
   [1, 0, 0]],
  [[0, 0, 0],
   [0, 0, 0]],
]),"L"),
(np.array([
  [[1, 1, 1],
   [1, 0, 1]],
  [[0, 0, 0],
   [0, 0, 0]],
]),"U")
]

board = np.array([
  [[1, 1, 1],
   [1, 1, 1]],
  [[1, 1, 1],
   [1, 1, 1]]
])

pre = pre_place(board, blocks, 4)
print(pre)
p_all = place_all(pre[0], [p[0] for p in pre[1]])
p_view = viewer(p_all)

for view in p_view:
  print_block(view)

all_parts = all_pattern([2,3,2], [5])
for parts in all_parts:
  print_block(parts)
print(len(all_parts))

all_blocks = [
(
[[[1, 1, 1],
  [0, 1, 0]],
 [[1, 0, 0],
  [0, 0, 0]]]
  , 2, "YT"
),(
[[[1, 1, 1],
  [1, 0, 0]],
 [[0, 0, 1],
  [0, 0, 0]]]
  , 2, "YL"
),(
[[[1, 1, 0],
  [0, 1, 0]],
 [[0, 0, 0],
  [0, 1, 0]]]
  , 3, "YV"
),(
[[[1, 1, 1],
  [1, 0, 1]],
 [[0, 0, 0],
  [0, 0, 0]]]
  , 2, "YU"
),(
[[[1, 1, 1],
  [0, 0, 1]],
 [[0, 0, 0],
  [0, 0, 1]]]
  , 2, "BL"
),(
[[[0, 1, 1],
  [1, 1, 0]],
 [[0, 0, 0],
  [1, 0, 0]]]
  , 2, "BS"
),(
[[[1, 1, 1],
  [0, 1, 1]],
 [[0, 0, 0],
  [0, 0, 0]]]
  , 3, "BP"
),(
[[[1, 1, 0],
  [1, 0, 0]],
 [[0, 0, 0],
  [0, 0, 0]]]
  , 4, "BV"
),(
[[[1, 1, 0],
  [1, 1, 0]],
 [[1, 0, 0],
  [0, 0, 0]]]
  , 3, "RO"
),(
[[[1, 1, 0],
  [1, 0, 0]],
 [[0, 0, 0],
  [1, 0, 0]]]
  , 3, "RV"
),(
[[[1, 1, 1],
  [0, 0, 1]],
 [[1, 0, 0],
  [0, 0, 0]]]
  , 2, "RL"
),(
[[[0, 1, 1],
  [1, 1, 0]],
 [[0, 0, 0],
  [0, 0, 0]]]
  , 2, "RZ"
),(
[[[1, 1, 0],
  [0, 1, 1]],
 [[1, 0, 0],
  [0, 0, 0]]]
  , 2, "GS"
),(
[[[1, 1, 1],
  [1, 0, 0]],
 [[0, 0, 0],
  [1, 0, 0]]]
  , 2, "GL"
),(
[[[1, 1, 1],
  [0, 1, 0]],
 [[0, 0, 0],
  [0, 0, 0]]]
  , 2, "GT"
),(
[[[1, 1, 1],
  [0, 0, 1]],
 [[0, 0, 0],
  [0, 0, 0]]]
  , 4, "GJ"
)]

problems = [[
  ["GLBPBVRO", "ROGJYTRZ", "GSYVRORV", "GLRVROGJ", "YVGJYTRO"],
  ["GJYVBVGL", "BVGTBLGJ", "BVRLYVRZ", "GJRZYTBV", "BLBVRVGJ"],
  ["YVBPBLRZ", "ROBSGTYV", "GTYLYVGL", "BVGSROYT", "GSRVRLYV"],
  ["YVGJROBL", "BLRORVGJ", "ROBVBLGL", "RVGJROGL", "RORLYVGJ"]
]]
problems = [[[[p[i:i+2] for i in range(0, 8, 2)] for p in ps] for ps in prs] for prs in problems]
problems = [pr for prs in problems for pr in np.transpose(prs, (1, 0, 2))]
problems = [[p for pr in prs for p in pr] for prs in problems]

all_blocks = [(np.array(b[0]), b[1], b[2]) for b in all_blocks]
def make_puzzles(block_num, sizes, num):
  r_maps = []
  for i in range(num):
    r_maps += [random_map(2, 5, 3, np.random.choice(sizes))]

  results = []
  for i, r_map in enumerate(r_maps):
    print("Make ", i)
    result = make_problem(all_blocks, r_map, block_num, 100)
    if len(result) >= 1:
      select, ans = result[0]
      for a in viewer(ans):
        print_block(a)
      for s in select:
        print(s[2])
        print_block(s[0])
      print_block(r_map)
      print("Find Answers: ", len(result))

      results += [(r_map, result)]
  return results

# res = make_puzzles(5, [22, 24], 10)
# write_pickle(res, "5block.pickle")

# res :: [
#   ( r_map
#   , results
#     [
#       ( select [
#           ( blocks, count, name )
#         ]
#       , ans [
#           ( ans, blocks )
#         ]
#       )
#     ]
#   )
# ]

res = read_pickle("5block.pickle")
print_problems(res)

max_parts = calc_max_parts(all_blocks)
print(res[0][1])
for r in res:
  print(ok_problem(r, problems, max_parts))

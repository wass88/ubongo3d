from random import sample
import numpy as np
from nputils import NpUtils
from itertools import combinations, product, chain

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

class Block:
  def __init__(self, block, name = ""):
    self.name = name
    self.block = block

  @staticmethod
  def from_list(block, name = ""):
    return Block(np.array(block), name)
  
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

  def place_all_pat(self, board, remain_min = 0):
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
            if p != None and p.min_connected() >= remain_min:
              res[(block, move)] = p
    return res # [(roted_block, rot)] = board
  
  def count(self):
    return np.sum(self.block)

class BlockList:
  def __init__(self, blocks):
    self.blocks = list(blocks)

  def replace(self, blocks):
    return BlockList(blocks)
  
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

  def get(self, name):
    for b in self.blocks:
      if b.name == name: return b
    return None
  
  def sublist(self, names):
    return self.replace([self.get(name) for name in names])
  
  def print(self):
    for block in self.blocks:
      print(block.name, block.count())
      block.print()

  def count(self):
    return sum(block.count() for block in self.blocks)

  def names(self):
    return list(b.name for b in self.blocks)

  def blockcount(self):
    names = self.names()
    return {name: names.count(name) for name in set(names)}

class Comb:
  @staticmethod
  def comb_choice(target, count, cond):
    dp = [ [] for _ in range(target + 1) ]
    dp[0] = [[]]

    for (w, n) in cond.items():
      for _ in range(1, n+1):
        for i in reversed(range(target - w + 1)):
            res = list(d + [w] for d in dp[i] if len(d) + 1 <= count)
            for r in res:
              if r not in dp[i + w]:
                dp[i + w] = dp[i + w] + [r]
    return (d for d in dp[target] if len(d) == count)

  def comb_dup(count, cond):
    # cond: Dict[int -> List[(Block, num)]]
    dp = [[] for _ in range(count + 1)]
    dp[0] = [[]]
    for (b, n) in cond:
      for _ in range(1, n+1):
        for i in reversed(range(count)):
          res = list(d + [b] for d in dp[i])
          for r in res:
            if r not in dp[i + 1]:
              dp[i + 1].append(r)
    return dp[count]
  
  @staticmethod
  def all_comb(target, count, cond):
    # target == Subset of cond Len = count
    # cond: Dict[int -> List[(Block, num)]]
    def all_blocks_num(blocks):
      return sum(map(lambda b: b[1], blocks))
    ns = dict((count, all_blocks_num(blocks)) for count, blocks in cond.items())
    chs = Comb.comb_choice(target, count, ns)
    res = []
    for ch in chs:
      r = ((count, ch.count(count)) for count in set(ch))
      r = (Comb.comb_dup(n, cond[c]) for c, n in r)
      r = product(*r)
      r = (chain.from_iterable(a) for a in r)
      res.append((i for i in k) for k in r)
    return chain.from_iterable(res)

def comb_test():
  #print(list(Comb.comb_choice(10, 3, {1:2, 2:2, 3:1, 4:1, 5:1})))
  #print(list(Comb.comb_choice(12, 3, {4:3})))
  #print(list(list(i) for i in Comb.comb_dup(3, [("A",2), ("B",3), ("C",1)])))
  #print(list(list(x) for x in Comb.all_comb(10, 3,
  #  {1: [("X1", 1)], 2: [("X2", 2)], 3:[("X3", 1)], 4:[("X", 1), ("Y", 2)], 5: [("A", 1), ("B", 3)]})))
  pass

class BlockCount:
  def __init__(self, blocks, counts):
    self.blocks = blocks
    self.counts = counts
    self.calc_block_sum()
  
  def replace(self, other):
    return BlockCount(self.blocks, other)

  def replace_from_list(self, blocks):
    counts = dict(self.counts)
    for name in counts.keys():
      counts[name] = 0
    for name, c in blocks.blockcount().items():
      counts[name] = c
    return self.replace(counts)

  def remove_list(self, blocks):
    counts = dict(self.counts)
    for name in blocks.names():
      counts[name] -= 1
      assert counts[name] >= 0
    return self.replace(counts)
  
  def is_include_list(self, blocks):
    counts = dict(self.counts)
    for name in blocks.names():
      counts[name] -= 1
    for v in counts.values():
      if v < 0: return False
    return True

  @staticmethod
  def from_block_list(blocks):
    sampleblocks = dict((b.name, b) for b in blocks.blocks)
    return BlockCount(sampleblocks, dict((b.name, 1) for b in blocks.blocks))

  def calc_block_sum(self):
    self.blockcounts = dict((name, block.count()) for name, block in self.blocks.items())
    counts = set(self.blockcounts.values())
    def countname(count):
      return [(self.blocks[name], self.counts[name]) for name, c in self.blockcounts.items() if c == count]
    self.countblocks = dict((count, countname(count)) for count in counts)

  def remove_list(self, list):
    counts = dict(self.counts)
    for block in list.blocks:
      counts[block.name] -= 1
      assert self.counts[block.name] >= 0
    return self.replace(counts)

  def all_comb_from_board(self, board, num):
    from ubongo import BlockList
    res = Comb.all_comb(board.count(), num, self.countblocks)
    return map(BlockList, res)
  
  def get_blocklist(self):
    from ubongo import BlockList
    return BlockList(self.blocks.values())

  def flat_counts(self):
    return sum( ([name] * num for name, num in self.counts.items()), [])

  def random_list(self, num):
    from ubongo import BlockList
    names = sample(self.flat_counts(), num)
    return BlockList( self.blocks[name] for name in names )

  def sum(self):
    return sum(self.counts.values())

  def split_blocks(self, players):
    num = self.sum() // players
    sf = self
    res = []
    for _ in range(players):
      r = sf.random_list(num)
      sf = sf.remove_list(r)
      r = self.replace_from_list(r)
      res.append(r)
    return res

  def calc_placeable(self, board):
    self.placeable = dict(
      (name, block.place_all_pat(board)) for name, block in self.blocks.items()
    )

_all_blocks = [
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
      
def all_blocks():
  _blocks = dict((name, Block.from_list(b, name)) for b, n, name in _all_blocks)
  return BlockCount(_blocks, dict((name, n) for b, n, name in _all_blocks))
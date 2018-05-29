import numpy as np

class NpUtils:
  X = 2
  Y = 1
  Z = 0

  @staticmethod
  def zeros(shape):
    return np.zeros(shape, dtype=np.int)

  @staticmethod
  def rotX(block):
    return np.flip(np.swapaxes(block, NpUtils.Y, NpUtils.Z), NpUtils.Z)

  @staticmethod
  def rotY(block):
    return np.flip(np.swapaxes(block, NpUtils.Z, NpUtils.X), NpUtils.X)

  @staticmethod
  def rotZ(block):
    return np.flip(np.swapaxes(block, NpUtils.X, NpUtils.Y), NpUtils.Y)

  @staticmethod
  def rotXYZ(block, x, y, z):
    res = block
    for i in range(x): res = NpUtils.rotX(res)
    for i in range(y): res = NpUtils.rotY(res)
    for i in range(z): res = NpUtils.rotZ(res)
    return res
    
  @staticmethod
  def lead_zero(a):
    res = 0
    for c in a:
      if c == 0: res += 1
      else: break
    return res
    
  @staticmethod
  def shirnk(block):
    zz = np.amax(np.amax(block, 2), 1)
    zy = np.amax(np.amax(block, 0), 1)
    zx = np.amax(np.amax(block, 0), 0)
    sz = NpUtils.lead_zero(zz)
    sy = NpUtils.lead_zero(zy)
    sx = NpUtils.lead_zero(zx)
    tz = NpUtils.lead_zero(reversed(zz))
    ty = NpUtils.lead_zero(reversed(zy))
    tx = NpUtils.lead_zero(reversed(zx))
    return block[sz:-tz, sy:-ty, sx:-tx]
  
  rotations = [
    (x, y, z) for x in range(4) for y in range(4) for z in range(4)
  ]

  @staticmethod
  def expand_merginal(board, num):
    return NpUtils.expand(board, 0, num, 0, num, 0, num)

  normal_rots = [(0, y, z) for y in range(4) for z in range(4)] + \
                [(1, 2 * y, z) for y in range(2) for z in range(4)]

  @staticmethod
  def expand(block, lz, rz, ly, ry, lx, rx):
    shape = block.shape
    block = np.concatenate((NpUtils.zeros([shape[NpUtils.Z], shape[NpUtils.Y], lx]),
                  block,
                  NpUtils.zeros([shape[NpUtils.Z], shape[NpUtils.Y], rx])), NpUtils.X)
    shape = block.shape
    block = np.concatenate((NpUtils.zeros([shape[NpUtils.Z], ly, shape[NpUtils.X]]),
                  block,
                  NpUtils.zeros([shape[NpUtils.Z], ry, shape[NpUtils.X]])), NpUtils.Y)
    shape = block.shape
    block = np.concatenate((NpUtils.zeros([lz, shape[NpUtils.Y], shape[NpUtils.X]]),
                  block,
                  NpUtils.zeros([rz, shape[NpUtils.Y], shape[NpUtils.X]])), NpUtils.Z)
    return block
  
  @staticmethod
  def expand_fit(inner, outer):
    boz, boy, box = outer.shape
    blz, bly, blx = inner.shape
    return NpUtils.expand(inner, 0, boz - blz, 0, boy - bly, 0, box - blx)
  
  @staticmethod
  def connected(block):
    block = np.copy(block)
    for z in range(len(block)):
      for y in range(len(block[0])):
        for x in range(len(block[0][0])):
          if block[z][y][x] != 0:
            NpUtils.remove_con(block, z, y, x)
            return np.max(block) == 0
    return True

  D6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
  @staticmethod
  def remove_con(block, z, y, x):
    if 0 <= z < len(block) and 0 <= y < len(block[0]) and 0 <= x < len(block[0][0]):
      if block[z][y][x] != 0:
        block[z][y][x] = 0
        for (dz, dy, dx) in NpUtils.D6:
          NpUtils.remove_con(block, z + dz, y + dy, x + dx)

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

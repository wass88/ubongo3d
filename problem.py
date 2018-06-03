import numpy as np
from random import choice, shuffle
from nputils import NpUtils
from board import Board

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

  def same_block(self, other):
    return sorted(self.blocks.names()) == sorted(other.blocks.names())

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
  def make_from_board(board, block_count, block_num, count = 10000000, rand = False):
    board_count = board.count()
    block_count = block_count.clone()
    block_count.calc_placeable(board)

    conds = list(block_count.all_comb_from_board(board, block_num))
    if rand: shuffle(conds)
    print("COND : ", len(conds))

    params = enumerate(map(lambda blocks: [blocks, board, block_count], conds))
    res = []
    for ib in params:
        problem = solve(ib)
        yield problem
        if problem:
            res.append(problem)

    print("FIND : ", len(res))

  def len(self):
    return len(self.problems)
  
  def append(self, problem):
    assert self.board == problem.board
    self.problems.append(problem)

  def extend(self, problemb):
    self.problems.extend(problemb.problems)
  
  def is_include(self, problem):
    return any(map(lambda p: p.same_block(problem),
      self.problems))

  def remove_index(self, index):
    del self.problems[index]
  
  def print_one(self):
    self.problems[0].print_ans()
  
  def set_name(self, name):
    self.name = name

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

    pre_search = 60
    pre_req = 2

    boards = []

    def make_uniq(blockc):
      if setting.block_uniq:
        return blockc.uniq_block()
      return blockc

    for i in range(player):
      print("PreSearch: ", i)
      ok = False
      while not ok:
        c = 0
        board = Board.random_board(setting.board_depth, setting.board_width,
                                   setting.board_height, choice(cond_size))
        probb = ProblemBoard.make_from_board(board, make_uniq(block_count), block_num, rand=True)
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
          probs = ProblemBoard.make_from_board(boards[j], make_uniq(counts), block_num, rand=True)
          prob = None
          for prob in probs:
            if prob:
              if res[j].is_include(prob):
                print("Same Problem")
                prob = None
                continue
              else:
                break
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
    
  def set_names(self, animal):
    for i, probb in enumerate(self.probboards):
      probb.name = animal + "-" + str(i)
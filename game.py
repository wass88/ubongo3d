from problem import Problem, ProblemBoard, ProblemSet

class GameSetting:
  def __init__(self, block_count, games, player, puzzles, block_num, block_uniq,
               board_nums, board_width, board_height, board_depth):
    self.block_count = block_count
    self.games = games
    self.player = player
    self.puzzles = puzzles
    self.block_num = block_num
    self.block_uniq = block_uniq
    self.board_nums = board_nums
    self.board_width = board_width
    self.board_height = board_height
    self.board_depth = board_depth

class Game:
  def __init__(self, probsets):
    self.probsets = probsets

  @staticmethod
  def make(setting):
    res = []
    for i in range(setting.games):
      s = ProblemSet.make(setting)
      res.append(s)
    return Game(res)

  def set_names(self, animals):
    for i, probset in enumerate(self.probsets):
      probset.set_names(animals[i]+" "+str(i))

  
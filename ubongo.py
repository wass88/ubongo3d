import numpy as np
from itertools import permutations, chain, combinations

from random import choice, shuffle
from nputils import NpUtils
from blocks import Block, BlockList, BlockCount, all_blocks
from board import Board
from problem import Answers, Problem, ProblemBoard, ProblemSet
from game import GameSetting, Game

from pickleutil import PickleUtil
from svg import SVG

def test_solver():
  alls = all_blocks()

  board = Board.random_board(2, 2, 4, 16)
  board.print()
  blocks = alls.get_blocklist().sublist(["BV", "BV", "BP", "BP"])
  problem = Problem(blocks, board)
  problem.print_ans()

def make_game():
  setting = GameSetting(
    block_count = all_blocks(),
    games = 1,
    player = 4,
    puzzles = 5,
    block_num = 5,
    board_nums = [21, 24],
    board_width = 5,
    board_height = 4,
    board_depth = 3
  )

  s = ProblemSet.make(setting)
  PickleUtil.write("data/game4player5block3depth", s)

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

  game = PickleUtil.read("data/game4player5block.2")
  animals = ["イグアナ",
              "カメ",
              "カメレオン",
              "コブラ",
              "コモドドラゴン",
              "トカゲ",
              "ヘビ",
              "ヤモリ",
              "アナコンダ",
              "ワニ"]
  game.set_names(animals)
  html = SVG.game(game)
  html.save("data/game.html")
    
if __name__ == "__main__":
  #test_solver()
  make_game()
  #solving()
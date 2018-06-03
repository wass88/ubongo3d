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
  alls = all_blocks()
  all_sub = alls.remove_list(alls.get_blocklist().sublist(["BV", "BV", "BV"]))
  print(all_sub.counts)
  block5depth2 = GameSetting(
    block_count = all_sub,
    games = 4,
    player = 4,
    puzzles = 5,
    block_num = 5,
    block_uniq = True,
    board_nums = [22, 24],
    board_width = 5,
    board_height = 4,
    board_depth = 2
  )
  block5depth3 = GameSetting(
    block_count = all_sub,
    games = 4,
    player = 4,
    puzzles = 5,
    block_num = 5,
    block_uniq = True,
    board_nums = [21, 24],
    board_width = 5,
    board_height = 3,
    board_depth = 3
  )
  block6depth2 = GameSetting(
    block_count = all_sub,
    games = 1,
    player = 4,
    puzzles = 5,
    block_num = 6,
    block_uniq = True,
    board_nums = [24, 26, 28],
    board_width = 5,
    board_height = 5,
    board_depth = 2
  )
  block4depth3 = GameSetting(
    block_count = all_sub,
    games = 4,
    player = 4,
    puzzles = 5,
    block_num = 4,
    block_uniq = True,
    board_nums = [18],
    board_width = 3,
    board_height = 4,
    board_depth = 3
  )

  s = Game.make(block4depth3)
  PickleUtil.write("data/game4player4block3depthuniq", s)

animals = {
  "reptile": ["イグアナ", "カメ", "カメレオン", "コブラ", "コモドドラゴン",
              "トカゲ", "ヘビ", "ヤモリ", "アナコンダ", "ワニ"],
  "fish": ["トビウオ", "ハゼ", "コバンザメ", "アンコウ", "ミノカサゴ",
           "ホウボウ", "マンボウ", "マグロ", "ウナギ", "エイ"],
  "cute": ["モモンガ", "ヤマネコ", "ウサギ", "カモシカ", "サル",
           "キツネ", "タヌキ", "クマ", "テン", "リス"],
  "bug": ["カゲロウ", "トンボ", "ナナフシ", "バッタ", "カマキリ",
          "カメムシ", "ヘビトンボ", "ノミ", "ハエ", "チョウ"],
  "bird": ["カッコウ", "カモ", "キジ", "キツツキ",  "コウノトリ", 
           "タカ", "ダチョウ", "チドリ", "ペリカン", "ペンギン"],
}

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

  game = PickleUtil.read("data/game4player4block3depthuniq")
  game.set_names(list(map(lambda n : "[3段] " + n, animals["fish"])))
  html = SVG.game_tate(game)
  html.save("data/game4.html")
    
if __name__ == "__main__":
  #test_solver()
  make_game()
  solving()
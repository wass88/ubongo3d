import pickle

class PickleUtil:
  @staticmethod
  def read(f):
    with open(f, mode="rb") as f:
      return pickle.load(f)

  @staticmethod
  def write(f, res):
    with open(f, mode="wb") as f:
      pickle.dump(res, f)

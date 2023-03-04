from typing import List

from allennlp.common import Registrable

from l2v.utils import py_utils


class Dataset(Registrable):
  """Dataset we can train/evaluate on"""

  def get_name(self) -> str:
    """Get the name of the dataset

    The name should by uniquely identified with the set of examples `load` returns since we might
    use it for caching.
    """
    raise NotImplementedError()

  def load(self) -> List:
    """Loads the examples"""
    raise NotImplementedError()


@Dataset.register('composite')
class CompositeDataset(Dataset):
  def __init__(self, datasets: List[Dataset], name):
    self.datasets = datasets
    self.name = name

  def get_name(self) -> str:
    return self.name

  def load(self) -> List:
    return py_utils.flatten_list(x.load() for x in self.datasets)


class CachingDataset(Dataset):
  def __init__(self, ds):
    self.ds = ds
    self._data = None

  def get_name(self) -> str:
    return self.ds.get_name()

  def load(self) -> List:
    if self._data is not None:
      return list(self._data)
    self._data = self.ds.load()
    return list(self._data)

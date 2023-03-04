import json
import logging
import pickle
import signal
import sys
import zlib
from collections import defaultdict
from json import JSONEncoder
from os import listdir, remove, walk, makedirs
from os.path import exists, join, isdir, basename, dirname, split, relpath
from shutil import rmtree
from typing import TypeVar, List, Iterable, Dict, Optional, Any, Tuple
import numpy as np
import requests
from allennlp.common.util import import_module_and_submodules
from scipy.optimize import minimize_scalar


def get_yes_no(msg):
  while True:
    txt = input(msg).strip().lower()
    if txt in {"y", "yes"}:
      return True
    if txt in {"n", "no"}:
      return False


def load_pickle_object(file_name):
  with open(file_name, "rb") as f:
    return pickle.load(f)


def dump_pickle_object(obj, file_name):
  with open(file_name, "wb") as f:
    pickle.dump(obj, f)


def load_json_object(file_name):
  with open(file_name, "r") as f:
    return json.load(f)


def flatten_probability_dist(x, thresh):
  if x.shape[0]*thresh < 1.0:
    raise ValueError("Not possible")

  def fn(alpha):
    return np.square(1.0 - np.minimum(x*alpha, thresh).sum())

  result = minimize_scalar(fn, bracket=[0.0, 1.0])
  if result.success:
    out = np.minimum(x*result.x, thresh)
    assert np.abs(out.sum() - 1.0) < 1e-4
    return out
  else:
    raise ValueError("Acceptable alpha not found")


class DelayedKeyboardInterrupt:
  """Context manager to delay keyboard interuprts"""

  def __enter__(self):
    self.signal_received = False
    self.old_handler = signal.signal(signal.SIGINT, self.handler)

  def handler(self, sig, frame):
    self.signal_received = (sig, frame)
    logging.info('SIGINT received. Delaying KeyboardInterrupt.')

  def __exit__(self, type, value, traceback):
    signal.signal(signal.SIGINT, self.old_handler)
    if self.signal_received:
      self.old_handler(*self.signal_received)


class DisableLogging:
  """Context manager the temporarily disables logging"""

  def __init__(self, to_level=logging.INFO):
    self.to_level = to_level

  def __enter__(self):
    self.prev_level = logging.root.manager.disable
    if self.prev_level < self.to_level:
      logging.disable(self.to_level)

  def __exit__(self, exc_type, exc_val, exc_tb):
    logging.disable(self.prev_level)


def dump_json_object(dump_object, file_name, indent=2):
  with open(file_name, "w") as f:
    json.dump(dump_object, f, indent=indent)


def add_stdout_logger():
  handler = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                datefmt='%m/%d %H:%M:%S', )
  handler.setFormatter(formatter)
  handler.setLevel(logging.DEBUG)

  root = logging.getLogger()

  # fiftyone adds an stdout logger for some reason, we detect here by looking for its
  # completely plain formatting (seriously why is it here?) and delete it
  for i, h in enumerate(list(root.handlers)):
    if h.formatter._fmt == '%(message)s':
      root.removeHandler(h)
  root.setLevel(logging.INFO)
  root.addHandler(handler)

  # Re-direction warning to  logging
  logging.captureWarnings(True)


def clear_if_nonempty(output_dir, override=False):
  if output_dir:
    if exists(output_dir) and listdir(output_dir):
      if override or get_yes_no("%s is non-empty, override (y/n)?" % output_dir):
        for x in listdir(output_dir):
          if isdir(join(output_dir, x)):
            rmtree(join(output_dir, x))
          else:
            remove(join(output_dir, x))
      else:
        raise ValueError(f"Output directory ({output_dir}) already exists and is not empty.")


def select_run_dir(run_dir):
  """If `run_dir` is top-level model dir with a single run, returns that run"""
  if exists(join(run_dir, "model.json")):
    candidates = []
    for filename in listdir(run_dir):
      filepath = join(run_dir, filename)
      if isdir(filepath) and filename.startswith("r"):
        candidates.append(filepath)
    if len(candidates) > 1:
      raise ValueError(f"Multiple runs in {run_dir}, please select one")
    elif len(candidates) == 0:
      raise ValueError(f"No runs found in {run_dir}")
    else:
      logging.info(f"Selecting run {basename(candidates[0])} for {run_dir}")
      run_dir = candidates[0]

  return run_dir


K = TypeVar('K')
T = TypeVar('T')


def transpose_list_of_dicts(lst: List[Dict[K, T]]) -> Dict[K, List[T]]:
  out = defaultdict(list)
  for r in lst:
    for k, v in r.items():
      out[k].append(v)
  return {k: v for k, v in out.items()}


def transpose_lists(lsts: Iterable[Iterable[T]]) -> List[List[T]]:
  """Transpose a list of lists."""
  return [list(i) for i in zip(*lsts)]


def sample_dict(x: Dict[K, T], sample: Optional[int]) -> Dict[K, T]:
  if sample is None or len(x) < sample:
    return x
  keys = list(x)
  to_keep = np.random.choice(keys, sample, replace=False)
  return {k: x[k] for k in to_keep}


def duration_to_str(seconds):
  sign_string = '-' if seconds < 0 else ''
  seconds = abs(int(seconds))
  days, seconds = divmod(seconds, 86400)
  hours, seconds = divmod(seconds, 3600)
  minutes, seconds = divmod(seconds, 60)
  if days > 0:
    return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
  elif hours > 0:
    return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
  elif minutes > 0:
    return '%s%dm%ds' % (sign_string, minutes, seconds)
  else:
    return '%s%ds' % (sign_string, seconds)


def flatten_list(iterable_of_lists: Iterable[Iterable[T]]) -> List[T]:
  """Unpack lists into a single list."""
  return [x for sublist in iterable_of_lists for x in sublist]


def val_to_str(val: float, fmt):
  if val is None:
    return "-"
  if isinstance(val, str):
    return val
  return fmt % (100*val)


def table_string(table: List[List[str]]) -> str:
  """Table as list-of=lists to string."""
  # print while padding each column to the max column length
  if len(table) == 0:
    return ""
  col_lens = [0] * len(table[0])
  for row in table:
    for i, cell in enumerate(row):
      col_lens[i] = max(len(cell), col_lens[i])

  formats = ["{0:<%d}" % x for x in col_lens]
  out = []
  for row in table:
    out.append(" ".join(formats[i].format(row[i]) for i in range(len(row))))
  return "\n".join(out)


def dict_of_dicts_as_table_str(data: Dict[str, Dict[str, Any]], val_format, all_keys=None,
                               top_right="_", table_format="even-spaced") -> str:
  """Table of row->col->value to string"""
  if all_keys is None:
    all_keys = {}
    for name, result in data.items():
      for key in result:
        if key not in all_keys:
          all_keys[key] = 0

  all_keys = list(all_keys)
  header = [top_right] + all_keys
  table = [header]
  for name, result in data.items():
    row = [name] + [val_to_str(result.get(x), val_format) for x in all_keys]
    table.append(row)
  if table_format == "even-spaced":
    return table_string(table)
  elif table_format == "none":
    return table
  elif table_format == "csv":
    return "\n".join(",".join(row) for row in table)
  elif table_format == "tsv":
    return "\n".join(",".join(row) for row in table)
  elif table_format == "latex":
    return "\n".join(" & ".join(row) + "\\\\" for row in table)
  else:
    raise ValueError()


def list_of_dicts_as_table_str(data: List[Dict[str, Any]], val_format,
                               all_keys=None, table_format="even-spaced") -> str:
  """Table of row->col->value to string"""
  if all_keys is None:
    all_keys = {}
    for result in data:
      for key in result:
        if key not in all_keys:
          all_keys[key] = 0

  all_keys = list(all_keys)
  header = all_keys
  table = [header]
  for result in data:
    row = [val_to_str(result.get(x), val_format) for x in all_keys]
    table.append(row)

  if table_format == "even-spaced":
    return table_string(table)
  elif table_format == "csv":
    return "\n".join(",".join(row) for row in table)
  else:
    raise ValueError()


def get_batch_bounds(n, n_batches):
  per_group = n // n_batches
  remainder = n % n_batches
  goup_sizes = np.full(n_batches, per_group, np.int)
  goup_sizes[:remainder] += 1

  batch_ends = np.cumsum(goup_sizes)

  assert batch_ends[-1] == n

  batch_starts = np.pad(batch_ends[:-1], [1, 0], "constant")
  bounds = np.stack([batch_starts, batch_ends], 1)
  return bounds


def int_to_str(k: int) -> str:
  if isinstance(k, int) and k % 1000 == 0:
    return str(k//1000) + "k"
  else:
    return str(k)


def balanced_merge_multi(lsts: Iterable[List]) -> List:
  """Merge lists while trying to keep them represented proportional to their lengths
  in any continuous subset of the output list
  """
  if len(lsts) == 0:
    raise ValueError("Given an empt list")
  if len(lsts) == 0:
    return lsts[0]
  lens = np.array([len(x) for x in lsts], dtype=np.float64)
  target_ratios = lens / lens.sum()
  current_counts = np.zeros(len(lsts), dtype=np.int32)
  out = []
  lsts = [list(x) for x in lsts]
  while True:
    if len(out) == 0:
      next_i = np.argmax(target_ratios)
    else:
      # keep a track of the current mixing ratio, and add in the most under-represented list
      # each step
      next_i = np.argmin(current_counts / len(out) - target_ratios)
    current_counts[next_i] += 1
    lst = lsts[next_i]
    out.append(lst.pop())
    if len(lst) == 0:
      target_ratios = np.delete(target_ratios, next_i)
      current_counts = np.delete(current_counts, next_i)
      lsts = lsts[:next_i] + lsts[next_i+1:]
      if len(lsts) == 0:
        break

  return out[::-1]


def nested_struct_to_flat(tensors, prefix=(), cur_dict=None) -> Dict[Tuple, Any]:
  """Converts a nested structure of dict/lists/tuples to a flat dict with tuple keys"""
  if cur_dict is None:
    cur_dict = {}
    nested_struct_to_flat(tensors, (), cur_dict)
    return cur_dict

  if isinstance(tensors, dict):
    if len(tensors) == 0:
      raise ValueError("Cannot convert empty dict")
    for k, v in tensors.items():
      if isinstance(k, int):
        # We currently use int keys to signal a list, so this would result in errors
        raise NotImplementedError("Integer keys")
      nested_struct_to_flat(v, prefix + (k, ), cur_dict)
  elif isinstance(tensors, (tuple, list)):
    if len(tensors) == 0:
      raise ValueError("Cannot convert empty tuples/lists")
    for ix, v in enumerate(tensors):
      nested_struct_to_flat(v, prefix + (ix, ), cur_dict)
  else:
    cur_dict[prefix] = tensors


def flat_to_nested_struct(nested: Dict):
  """Undos the effect of `nested_struct_to_flat`"""
  if len(nested) == 0:
    return None
  if isinstance(next(iter(nested.keys()))[0], str):
    out = {}
  else:
    out = []

  for prefix, value in nested.items():
    parent = out
    for i, key in enumerate(prefix[:-1]):
      next_parent = {} if isinstance(prefix[i+1], str) else []
      if isinstance(key, str):
        if key not in parent:
          parent[key] = next_parent
        parent = parent[key]

      elif isinstance(key, int):
        if len(parent) < key + 1:
          parent += [None] * (key + 1 - len(parent))
        if parent[key] is None:
          parent[key] = next_parent
        parent = parent[key]

      else:
        raise NotImplementedError()

    key = prefix[-1]
    if isinstance(key, int):
      if len(parent) < key + 1:
        parent += [None] * (key + 1 - len(parent))
    parent[prefix[-1]] = value

  return out


IMPORT_DONE = False


def import_all():
  global IMPORT_DONE
  if not IMPORT_DONE:  # TODO does this guard have a purpose?
    for module in ["model"]:
      import_module_and_submodules(f"l2v.{module}")
      IMPORT_DONE = True


def is_model_dir(x):
  return exists(join(x, "model.json"))


def is_run_dir(x, require_done):
  if exists(join(x, "status.json")):
    if not require_done:
      return True
    else:
      return load_json_object(join(x, "status.json"))["done"]
  return False


def extract_runs(model_dir, require_done=True):
  runs = []
  for run_dir in listdir(model_dir):
    run_dir = join(model_dir, run_dir)
    if is_run_dir(run_dir, require_done):
      runs.append(run_dir)
  return runs


# def transpose_dict_of_dicts(data: Dict[Any, Dict[K, T]]) -> Dict[k, Dict[Any, T]]:
#   out = defaultdict(dict)
#   for k1, v1 in data.items():
#     for k2, v2 in v1.items():
#       out[k2][k1] = v2
#   return {k: v for k, v in out.items()}


def find_models(roots, require_runs=True, require_done=True) -> Dict[str, Tuple[str, List[str]]]:
  """Find all trained models in a directory, or list of directories

  #:return A dictionary of name -> (model_dir, runs) of models found in `roots`. The name
  is derived from the location of model_dir relate to the input root.
  """

  if isinstance(roots, str) and is_run_dir(roots, require_done):
    return {split(roots)[1]: (dirname(roots), [roots])}

  if isinstance(roots, str):
    roots = [(None, roots)]
  elif isinstance(roots, dict):
    roots = list(roots.items())
  elif len(roots) == 1:
    roots = [(None, roots[0])]
  else:
    names = [x.rstrip("/").split("/")[-2] for x in roots]
    roots = list(zip(names, roots))

  models = {}
  for root_name, root in roots:
    if is_model_dir(root):
      runs = []
      for run_dir in listdir(root):
        run_dir = join(root, run_dir)
        if is_run_dir(run_dir, require_done):
          runs.append(run_dir)
      model_name = basename(root)
      if root_name:
        model_name = join(root_name, model_name)
      models[model_name] = (root, runs)
      continue

    for dirpath, dirnames, filenames in walk(root):
      for model_dir in dirnames:
        model_dir = join(dirpath, model_dir)
        if not is_model_dir(model_dir):
          continue

        model_name = relpath(model_dir, root)
        if root_name:
          model_name = join(root_name, model_name)

        runs = extract_runs(model_dir, require_done)
        if not require_runs or len(runs) > 0:
          models[model_name] = (model_dir, runs)

  return models


def consistent_hash(text):
  text = text.encode("utf-8")
  return hex(zlib.crc32(text)) + "-" + hex(zlib.crc32(b'1' + text[::-1]))


def ensure_dir_exists(filename):
  """Make sure the parent directory of `filename` exists"""
  makedirs(dirname(filename), exist_ok=True)


def download_to_file(url, output_file):
  """Download `url` to `output_file`, intended for small files."""
  logging.info(f"Downloading file from {url} to {output_file}")
  ensure_dir_exists(output_file)
  with requests.get(url) as r:
    r.raise_for_status()
    with open(output_file, 'wb') as f:
      f.write(r.content)


class NumpyArrayEncoder(JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return JSONEncoder.default(self, obj)


def subsample(out, num_examples, seed=613423):
  if num_examples is None:
    return out
  out.sort(key=lambda x: x.example_id)
  if isinstance(num_examples, float):
    sample = int(round(len(out)*num_examples))
  else:
    sample = num_examples
  np.random.RandomState(seed).shuffle(out)
  return out[:sample]

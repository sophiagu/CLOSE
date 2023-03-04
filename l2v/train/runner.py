

# This needs to be a top-level class so it can be distributed
import json
from collections import Callable
from dataclasses import dataclass, asdict
from os.path import join, dirname
from typing import Dict, Union

import h5py
import torch
from allennlp.common import FromParams, Params
from torch.utils.data import DataLoader
from tqdm import tqdm

from l2v.model.load_model import load_model
from l2v.model.model import ExampleOutput, PredictionArg, Model
from l2v.utils import pytorch_utils, py_utils
from l2v.utils.py_utils import load_json_object
from l2v.utils.pytorch_utils import QueueDataset
from l2v.utils.to_params import to_params_any


def prediction_args_to_json(prediction_args):
  prediction_args_dict = {}
  for k, v in prediction_args.items():
    if isinstance(v, FromParams):
      v = to_params_any(v, Union[PredictionArg, float, int, str])
    prediction_args_dict[k] = v
  return prediction_args_dict


def save_example_output(output: Dict[str, ExampleOutput], output_dir):
  predictions = {}
  for key, out in output.items():
    predictions[key] = dict(
      answer=out.text,
      probs=None if out.text_logprobs is None else out.text_logprobs.tolist()
    )

  with open(output_dir + "/predictions.json", "w") as f:
    json.dump(predictions, f)


@dataclass
class CollateWithBatch(Callable):
  collate: Callable

  def __call__(self, batch):
    return batch, self.collate(batch)


@dataclass
class DataLoaderBuilder(FromParams):
  """Construct a DataLoader

  Essentially a wrapper around the extra parameters a DataLoader needs
  """
  batch_size: int
  num_workers: int = 0
  pin_memory: bool = True
  prefetch_factor: int = 2
  persist_workers: bool = False

  def build(self, dataset, collate, sampler=None, shuffle=False,
            batch_size=None, batch_sampler=None):
    return DataLoader(
      dataset,
      batch_size=batch_size if batch_size is not None else self.batch_size,
      collate_fn=collate,
      num_workers=self.num_workers,
      shuffle=shuffle,
      batch_sampler=batch_sampler,
      sampler=sampler,
      pin_memory=self.pin_memory,
      prefetch_factor=self.prefetch_factor,
      persistent_workers=self.persist_workers
    )


def build_per_example_output(examples, output, beams_to_keep=1):
  out = {}
  for ex, ex_out in zip(examples, output):
    out[ex.get_example_id()] = ex_out.set_beams_to_keep(beams_to_keep)
  return out


def run(model_source, examples, device,
        batch_size, num_workers, prediction_args, beams_to_keep=1,
        desc="eval", nopbar=False):
  if len(set(ex.get_example_id() for ex in examples)) != len(examples):
    raise ValueError("Repeated ids in examples")
  if isinstance(device, list):
    return run_dist(
      model_source, examples, device,
      batch_size, num_workers, prediction_args, beams_to_keep=beams_to_keep,
      desc=desc, nopbar=nopbar
    )
  else:
    model = load_model(model_source, device=device)
    model.set_prediction_args(**prediction_args)

    loader = DataLoader(
      [model.preprocess_example(x) for x in examples],
      batch_size=batch_size,
      collate_fn=CollateWithBatch(model.get_collate()),
      num_workers=num_workers,
      shuffle=False,
      pin_memory=True
    )

    return run_model(model, loader, beams_to_keep, desc, nopbar,
                     prediction_args=prediction_args)


def run_model(
    model, data_loader, beams_to_keep=1,
    desc="eval", nopbar=False, model_device=None,
    prediction_args=None
) -> Dict[str, ExampleOutput]:
  if prediction_args is None:
    prediction_args = {}
  model.eval()
  if model_device is None:
    model_device = pytorch_utils.get_model_device(model)

  model.set_prediction_args(**prediction_args)

  if desc is None:
    desc = "eval"

  out = {}
  if nopbar:
    it = data_loader
  else:
    it = tqdm(data_loader, desc=desc, ncols=100)

  for examples, batch in it:
    batch = pytorch_utils.to_device(batch, model_device)
    with torch.no_grad():
      output = model.predict(**batch)

    out.update(build_per_example_output(examples, output, beams_to_keep))

  return out


def run_dist(model_source, examples, devices,
             batch_size, num_workers, prediction_args, beams_to_keep=1,
             desc="eval", nopbar=False):
  ctx = torch.multiprocessing.get_context("spawn")

  params = Params(load_json_object(join(dirname(model_source), "model.json")))
  py_utils.import_all()
  with py_utils.DisableLogging():
    model: Model = Model.from_params(params)
  model.set_prediction_args(**prediction_args)  # In case this effects `preprocess_example`

  input_q = ctx.Queue()
  for ex in examples:
    # Putting tensors into the queue can lead to tricky errors as it hits
    # torch's tensor-sharing code. It seems easier to just work-around it
    # since we aren't worried about memory consumption anyway
    assert all(not isinstance(v, torch.Tensor) for v in asdict(ex).values())
    # Block so the queue will be completely filled before
    # multiprocessing start
    input_q.put(model.preprocess_example(ex), block=True)

  out_q = ctx.Queue()

  args = (
    model_source, devices, input_q, out_q, batch_size,
    num_workers, beams_to_keep, prediction_args)

  context = torch.multiprocessing.spawn(_run_worker, nprocs=len(devices), args=args, join=False)

  pbar = tqdm(desc=desc, disable=nopbar, total=len(examples), ncols=100)
  out = {}
  while True:
    item = out_q.get()
    if isinstance(item, Exception):
      # A worker has failed, call .join immediately to get the exception and stacktrace
      # We need to use `cancel_join_thread` first so the child process terminates correctly
      # even though the queues still has enqueud data
      input_q.cancel_join_thread()
      out_q.cancel_join_thread()
      while not context.join():
        pass
      raise RuntimeError("context.join() should have raised an error!")

    for k, v in item.items():
      assert k not in out
      out[k] = v
      pbar.update(1)

    if pbar.n == len(examples):
      assert input_q.empty()
      break
    assert pbar.n < len(examples)

  while not context.join():
    pass

  return out


def _run_worker(
    rank, model, devices, queue, out_q, batch_size, num_workers,
    beams_to_keep, prediction_args):
  try:
    device = devices[rank]
    with py_utils.DisableLogging():
      model: Model = load_model(model, device=device)
    model.set_prediction_args(**prediction_args)
    dataset = QueueDataset(queue)
    loader = DataLoader(
      dataset, batch_size=batch_size, num_workers=num_workers,
      collate_fn=CollateWithBatch(model.get_collate()))
    for examples, batch in loader:
      batch = pytorch_utils.to_device(batch, device)
      with torch.no_grad():
        output = model.predict(**batch)
      if beams_to_keep is not None:
        out_q.put(build_per_example_output(examples, output, beams_to_keep))
  except Exception as e:
    out_q.put(e)
    raise e

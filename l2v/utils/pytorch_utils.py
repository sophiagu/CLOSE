import logging
import re
from queue import Empty
from typing import Union

from torch.nn import functional as F
import torch
from torch import nn
from torch.utils.data import IterableDataset
from transformers.models.gptj import modeling_gptj


def get_device(device_name: Union[None, str, int]=None):
  if device_name is None:
    if torch.cuda.is_available():
      logging.info("cuda found, defaulting to cuda")
      return torch.device('cuda')
    else:
      logging.info("cuda not found, using cpu")
      return torch.device('cpu')
  else:
    try:
      device_name = int(device_name)
    except ValueError:
      pass
    return torch.device(device_name)


def to_device(batch, device):
  if batch is None:
    return None
  if isinstance(batch, (float, int, str)):
    return batch
  if isinstance(batch, dict):
    return {sub_k: to_device(sub_v, device) for sub_k, sub_v in batch.items()}
  if isinstance(batch, (tuple, list)):
    return [to_device(x, device) for x in batch]
  else:
    return batch.to(device)


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
  dim = x.shape[-1]
  if seq_len is None:
    seq_len = x.shape[seq_dim]
  inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device) / dim))
  sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len, device=x.device), inv_freq).float()
  return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def patch_gptj():
  modeling_gptj.fixed_pos_embedding = fixed_pos_embedding


def get_devices(devices):
  if isinstance(devices, list) and len(devices) > 1:
    out = []
    for x in devices:
      try:
        out.append(int(x))
      except ValueError:
        out.append(x)
    return out

  if isinstance(devices, list):
    devices = devices[0]

  if devices is not None:
    try:
      return int(devices)
    except ValueError:
      return devices
  else:
    if torch.cuda.is_available():
      logging.info("cuda found, defaulting to cuda")
      return 'cuda'
    else:
      logging.info("cuda not found, using cpu")
      return 'cpu'


def get_model_device(module: torch.nn.Module):
  return next(module.parameters()).device


def seq_len_to_binary_mask(seq_len, max_len=None):
  if max_len is None:
    max_len = seq_len.max()
  return seq_len.unsqueeze(1) > torch.arange(0, max_len, device=seq_len.device).unsqueeze(0)


def segment_mean(x, segments):
  """
  :param x: [batch, dim]
  :param segments: [batch]
  :return: [n_segments, dim]
  """
  counts = torch.unique_consecutive(segments.cpu(), return_counts=True)[1]
  start = 0
  means = []
  for c in counts:
    means.append(x[start:start+c].mean(0))
    start += c
  return torch.stack(means, 0)


def concat_masked_sequences(
    seq1, mask1,
    seq2, mask2
):
  batch = seq1.size(0)
  if mask1 is None and mask2 is None:
    return torch.cat([seq1, seq2], 1), None
  if mask1 is None:
    if len(mask2.size()) == 1:
      raise NotImplementedError("Sequence length masks2")
    out = torch.cat([seq1, seq2], 1)
    mask = torch.cat([
      torch.ones(batch, seq1.size(1), device=seq1.device, dtype=mask2.dtype),
      mask2
    ], 1)
    return out, mask
  elif mask2 is None:
    seq2_len = seq2.size(1)

    if len(mask1.size()) == 2:
      assert mask1.dtype == torch.bool or torch.all(torch.logical_or(mask1 == 0, mask1 == 1))
      seq_len1 = mask1.int().sum(1)
    else:
      assert mask1.dtype == torch.long and len(mask1.size()) == 1
      seq_len1 = mask1

    out = F.pad(seq1, [0, 0, 0, seq2_len, 0, 0])
    for i in range(batch):
      out[i, seq_len1[i]:seq_len1[i]+seq2_len] = seq2[i]
    return out, seq_len_to_binary_mask(seq_len1 + seq2_len)
  else:
    # both mask are not none
    if len(mask1.size()) != 1:
      raise NotImplementedError("Binary mask1")
    else:
      seq_len1 = mask1

    if len(mask2.size()) == 2:
      assert mask2.dtype == torch.bool or torch.all(torch.logical_or(mask2 == 0, mask2 == 1))
      seq_len2 = mask2.int().sum(1)
    else:
      seq_len2 = mask2

    out_len = (seq_len1 + seq_len2).max()
    to_pad = out_len - seq1.size(1)
    out = F.pad(seq1, [0, 0, 0, to_pad, 0, 0])
    for i in range(batch):
      out[i, seq_len1[i]:seq_len1[i]+seq_len2[i]] = seq2[i, :seq_len2[i]]
    return out, seq_len_to_binary_mask(seq_len1 + seq_len2)


class QueueDataset(IterableDataset):
  def __init__(self, q):
    """
    q: Queue with all elements we want to yield already queue up
    """
    self.q = q

  def __iter__(self):
    while True:
      try:
        item = self.q.get(block=False)
        if item is None:
          return  # Allow None to also signal the end of the dataset
        yield item
      except Empty:
        # Even for non-blocking calls, it looks like empty can be raised even if the queue
        # had elements in it (due to locking issues?) double check here
        if self.q.empty():
          return


def replace_parameters(model: nn.Module, persistent):
  """Replace's the model parameters with buffers, useful
  from frozen model, or to keep parameters out of state dict
  by setting persistent to False
  """
  for child in model.modules():
    for name, param in list(child.named_parameters(recurse=False)):
      child.__delattr__(name)
      child.register_buffer(name, param.data, persistent)


def rotate_towards(x, target, target_dot):
  """
  :param x: [batch, dim] or [1, dim] unit vectors to move
  :param target: [batch, dim], [1, dim] or [dim] unit vector targets
  :param target_dot: [batch], between -1 and 1

  Rotates `x` to have a cosine distance of `target_dot` with `target`. More precisely:
  Find the unit vector closest to x on the plane defined by the points (x, target) that has a
  cosine distance between x and target of `target_dot`. If x, target are co-linear,
  or nearly co-linearly, `x` is returned instead.
  """
  if len(target.size()) == 1:
    target = target.unsqueeze(0)

  # The quadratic equations solves using the square of `target_dot`, so it
  # is insensitive to the sign of `target_dot`
  # To handle this we solve the equivalent but reflected problem  if `target_dot` is
  # negative, and then reflect the solution
  is_negative = ((target_dot >= 0).float()*2-1).unsqueeze(1)
  x = x * is_negative

  c_sq = target_dot*target_dot
  xy = (x * target).sum(-1)
  x_sub_y = x - target
  x_sub_y_sq = (x_sub_y * x_sub_y).sum(-1)
  a = ((xy - 1)**2 - c_sq*x_sub_y_sq)
  b = 2*(xy - 1)*(1 - c_sq)
  c = 1 - c_sq

  # The alternative solution to the quadratic is never desired according to
  # my testing, although honesty I am not sure why
  delta = (-b - torch.sqrt(b*b - 4*a*c)) / (2 * a)
  delta = delta.unsqueeze(-1)
  out = x*delta + target * (1-delta)
  norm = out.norm(dim=-1, keepdim=True)

  # Norm too small means x/target are colinear or amost colinear, we just give up and return
  # x here since the problem becomes ill-defined
  out = torch.where(norm > 1e-8, out / norm, x)
  out = out * is_negative
  return out

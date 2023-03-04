from os.path import join, dirname
from typing import List, Dict, Any

import torch
from allennlp.common import Registrable, FromParams
from torch import nn

from l2v.utils.to_params import to_params


class Layer(nn.Module, Registrable):
  pass


ACTIVATIONS = {
  "relu": nn.ReLU,
  "tanh": nn.Tanh
}


# Register some common torch Modules as Layers
@Layer.register("linear")
class Linear(Layer, nn.Linear):

  def _to_params(self) -> Dict[str, Any]:
    return dict(in_features=self.in_features, out_features=self.out_features,
                bias=self.bias is not None)


@Layer.register("add-bias")
class AddBias(Layer):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.bias = nn.Parameter(torch.zeros(dim))

  def forward(self, x):
    return x + self.bias


@Layer.register("seq")
class Sequential(Layer, nn.Sequential):
  def __init__(self, args: List[Layer]):  # TODO currently don't support *args
    super(Sequential, self).__init__(*args)

  def _to_params(self) -> Dict[str, Any]:
    return dict(args=[to_params(x, Layer) for x in self])


@Layer.register("mlp")
class MLP(Layer):

  def __init__(self, layer_sizes: List[int], activation, final_activation=None,
               dropout=0.0):
    super().__init__()
    self.final_activation = final_activation
    self.activation = activation
    self.layer_sizes = layer_sizes
    self.dropout = dropout

    parts = []
    for ix in range(len(layer_sizes)-1):
      size_from, size_to = layer_sizes[ix:ix+2]
      if ix != 0 and self.dropout:
        parts.append(nn.Dropout(self.dropout))
      parts.append(nn.Linear(size_from, size_to))
      if self.activation:
        parts.append(ACTIVATIONS[self.activation]())
    if self.final_activation:
      parts.append(ACTIVATIONS[self.final_activation]())
    self.mlp = nn.Sequential(*parts)

  def forward(self, x):
    return self.mlp(x)

import json
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Any, Optional

from l2v import file_paths
from l2v.data.dataset import Dataset
from l2v.data.vqa_v2 import VqaExample
from l2v.train.evaluator import PerExampleEvaluator, Evaluator
from l2v.utils import py_utils
from l2v.utils.py_utils import int_to_str

import numpy as np


@dataclass
class VisualEntailmentExample:
  example_id: str
  image_id: Optional[str]
  label: str
  hypothesis: str
  premise: str

  def get_example_id(self):
    return self.example_id


@Evaluator.register("entailment")
class EntailmentEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[VisualEntailmentExample], predictions: Dict[str, Any]):
    scores = []
    for ex in examples:
      text = predictions[ex.example_id].text[0]
      scores.append(dict(accuracy=text==ex.label))
    return scores


@Dataset.register("mnli")
class MNLI(Dataset):
  def __init__(self, split, sample=None):
    self.split = split
    self.sample = sample

  def load(self) -> List:
    src = f"multinli_1.0_{self.split}.txt"
    out = []
    with open(join(file_paths.MNLI, src)) as f:
      f.readline()
      for line in f:
        parts = line.split("\t")
        label = parts[0]
        hypothesis, premise = parts[5:7]
        example_id = parts[9]
        out.append(VisualEntailmentExample(
          example_id=example_id,
          premise=premise.strip(),
          image_id=None,
          label=label,
          hypothesis=hypothesis.strip(),
        ))
    return py_utils.subsample(out, self.sample)


@Dataset.register("snli-ve")
class VisualEntailment(Dataset):

  def __init__(self, split, sample=None, use_images=True):
    self.split = split
    self.sample = sample
    self.use_images = use_images

  def get_name(self) -> str:
    if self.use_images:
      text = "snli"
    else:
      text = "snli-ve"
    text += f"-{self.split}"
    if self.sample is not None:
      text += f"-s{int_to_str(self.sample)}"
    return text

  def load(self):
    out = []
    split = self.split
    if split == "val":
      split = "dev"
    src = join(file_paths.SNLI_VE_HOME, f"snli_ve_{split}.jsonl")
    with open(src) as f:
      lines = f.readlines()

    if self.sample is not None:
      np.random.RandomState(132124).shuffle(lines)
      lines = lines[:self.sample]

    for line in lines:
      example = json.loads(line)
      image_id = "flicker30k/" + example["Flickr30K_ID"] + ".jpg"
      out.append(VisualEntailmentExample(
        example_id="snli-ve/" + example["pairID"],
        image_id=image_id if self.use_images else None,
        label=example["gold_label"],
        hypothesis=example["sentence2"],
        premise=example["sentence1"]
      ))
    return out


if __name__ == '__main__':
  data = MNLI("train").load()
  print(data[0])
  # print(np.mean([len(x.question) for x in data]))
  # print(np.mean([len(x.image_text) for x in data]))

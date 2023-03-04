import json
import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from os.path import join
from typing import Optional, Dict, Any, List

from l2v import file_paths
from l2v.data.dataset import Dataset
from l2v.utils import image_utils, py_utils
from l2v.utils.py_utils import int_to_str, load_json_object
import numpy as np


@dataclass(frozen=True)
class CaptioningExample:
  example_id: str
  image_id: Optional[str]
  captions: List[str]
  meta: Optional[Dict[str, Any]] = None

  @property
  def crop(self):
    return None

  def get_example_id(self):
    return self.example_id


ANNOTATION_FILE_MAP = {
  "train": "captions_train2014.json",
  "val": "captions_val2014.json"
}


def _load(caption_file, sample=None):
  logging.info(f"Loading captioning data from {caption_file}")
  with open(caption_file) as f:
    data = json.load(f)

  subset = caption_file.split("_")[1].split(".")[0]
  assert subset in {"train2014", "val2014"}

  image_id_to_cap = defaultdict(list)
  for anno in data["annotations"]:
    image_id_to_cap[anno["image_id"]].append(anno)

  image_ids = image_id_to_cap
  if sample:
    image_ids = sorted(image_ids)
    if isinstance(sample, float):
      sample = int(round(len(image_ids)*sample))
    np.random.RandomState(613423).shuffle(image_ids)
    image_ids = image_ids[:sample]

  out = []
  for image_id in image_ids:
    caps = image_id_to_cap[image_id]
    cap_objects = []
    for cap in caps:
      cap_objects.append(cap['caption'])

    out.append(CaptioningExample(
      f"coco-cap-{image_id}",
      image_utils.get_coco_image_id(subset, image_id),
      cap_objects
    ))

  return out


@Dataset.register("coco-cap")
class CocoCaptioning(Dataset):

  def __init__(self, split, sample=None):
    if split not in ANNOTATION_FILE_MAP:
      raise ValueError()
    self.split = split
    self.sample = sample

  def get_name(self) -> str:
    name = f"coco-cap-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List[CaptioningExample]:
    return _load(join(file_paths.COCO_ANNOTATIONS, ANNOTATION_FILE_MAP[self.split]), self.sample)


def load_gpv_captioning(file, sample=None) -> List[CaptioningExample]:
  """Load GPV-I captioning data"""
  logging.info(f"Loading data from {file}")
  raw_instances = load_json_object(file)
  grouped_by_image = defaultdict(list)
  for i, x in enumerate(raw_instances):
    meta = {}
    if "coco_categories" in x:
      cats = x["coco_categories"]
      meta.update({
        "gpv1-unseen": cats["unseen"],
        "gpv1-seen": cats["seen"],
      })
    if "answer" in x:
      meta["gpv1-answer"] = x["answer"]
    meta["gpv1-query"] = x["query"]
    grouped_by_image[image_utils.get_coco_image_id(x["image"]["subset"], x["image"]["image_id"])].append(x.get("answer"))

  image_ids = list(grouped_by_image)
  if sample:
    image_ids.sort()
    np.random.RandomState(97982).shuffle(image_ids)
    image_ids = image_ids[:sample]

  out = []
  for image_id in image_ids:
    captions = grouped_by_image[image_id]
    _, subset, image_file = image_id.split("/")
    image_id_int = image_file.split("_")[-1].split(".")[0]
    out.append(CaptioningExample(f"coco-cap-{subset}-{image_id_int}", image_id, captions))
  return out


@Dataset.register("coco-cap-noimg")
class CocoCaptioningNoImages(Dataset):

  def __init__(self, split="train", sample=None, percent=None):
    self.sample = sample
    self.split = split
    self.percent = percent

  def get_name(self) -> str:
    name = f"coco-cap-noimages-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List[CaptioningExample]:
    out = _load(join(file_paths.COCO_ANNOTATIONS, ANNOTATION_FILE_MAP[self.split]), self.sample)
    if self.percent is None:
      n = len(out)
    else:
      np.random.RandomState(149504).shuffle(out)
      n = int(len(out) * self.percent)
    return out[:n] + [replace(x, image_id=None) for x in out[n:]]


@Dataset.register("coco-cap-ungrouped")
class CocoCaptioningUngrouped(Dataset):

  def __init__(self, split="train", sample=None, n_to_select=1):
    self.sample = sample
    self.split = split
    self.n_to_select = n_to_select

  def get_name(self) -> str:
    name = f"flattened-captions-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List[CaptioningExample]:
    out = _load(join(file_paths.COCO_ANNOTATIONS, ANNOTATION_FILE_MAP[self.split]))
    text = py_utils.flatten_list([x.captions for x in out])
    np.random.RandomState(63242).shuffle(text)
    if self.sample:
      text = text[:self.sample]
    return [CaptioningExample(str(i), None, captions=[x]) for i, x in enumerate(text)]


@Dataset.register("coco-sce")
class CocoSCE(Dataset):

  def __init__(self, split: str, sample: Optional[int]=None, include_images=True):
    self.split = split
    self.sample = sample
    self.include_images = include_images

  def get_name(self) -> str:
    name = f"coco-sce-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List:
    is_ood = False
    if self.split == "train":
      f_name = f"{self.split}.json"
    elif self.split == "val":
      f_name = f"val.json"
    elif self.split in {"ood-test", "ood-val"}:
      f_name = f"test.json"
      is_ood = True
    else:
      raise NotImplementedError(self.split)
    data = load_gpv_captioning(join(file_paths.COCO_SCE_HOME, f_name), self.sample)

    if not self.include_images:
      data = [replace(x, image_id=None) for x in data]

    if is_ood:
      np.random.RandomState(23523).shuffle(data)
      n = len(data) // 2
      if self.split == "ood-test":
        data = data[n:]
      elif self.split == "ood-val":
        data = data[:n]
      else:
        raise RuntimeError()

    return data


if __name__ == '__main__':
  py_utils.add_stdout_logger()
  print(len(CocoSCE("ood-val").load()))
from dataclasses import dataclass
from os.path import join
from typing import List, Optional
import numpy as np

from collections import Counter

from l2v import file_paths
from l2v.data.dataset import Dataset
from l2v.utils import image_utils
from l2v.utils.py_utils import int_to_str, load_json_object

ANNOTATION_FILE_MAP = {
  "train": [
    "v2_OpenEnded_mscoco_train2014_questions.json",
    "v2_mscoco_train2014_annotations.json"
  ],
  "val": [
    "v2_OpenEnded_mscoco_val2014_questions.json",
    "v2_mscoco_val2014_annotations.json"
  ]
}


@dataclass
class VqaExample:
  example_id: str
  question: str
  image_id: str
  question_type: str
  answers: Counter
  image_text: str
  multiple_choice_answer: str=None
  answer_type: Optional[str]=None

  def get_example_id(self):
    return self.example_id


def _load(q_file, a_file, sample, subset):
  q_data = load_json_object(q_file)["questions"]
  a_data = load_json_object(a_file)

  if sample:
    q_data = sorted(q_data, key=lambda x: x["question_id"])
    q_data = np.random.RandomState(613423).choice(q_data, sample, replace=False)

  anno_map = {}
  for anno in a_data["annotations"]:
    anno_map[anno["question_id"]] = anno

  out = []
  for q in q_data:
    anno = anno_map[q["question_id"]]
    image_id = image_utils.get_coco_image_id(subset, q["image_id"])
    out.append(VqaExample(
      q["question_id"], q["question"], image_id, anno["question_type"],
      Counter(x["answer"] for x in anno["answers"])
    ))
  return out


@Dataset.register("vqa-v2")
class Vqa2(Dataset):

  def __init__(self, split, sample=None):
    if split not in ANNOTATION_FILE_MAP:
      raise ValueError()
    self.split = split
    self.sample = sample

  def get_name(self) -> str:
    name = f"vqa-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List[VqaExample]:
    q_file, a_file = ANNOTATION_FILE_MAP[self.split]
    q_file = join(file_paths.VQA_ANNOTATIONS, q_file)
    a_file = join(file_paths.VQA_ANNOTATIONS, a_file)
    return _load(q_file, a_file, self.sample, self.split + "2014")


if __name__ == '__main__':
  print(len(Vqa2("val").load()))
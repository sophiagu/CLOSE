import zlib
from os import listdir
from os.path import join, exists

import h5py
import imagesize
import torch

from l2v import file_paths
from l2v.file_paths import CLIP_VECTOR_CACHE
from l2v.utils import py_utils

_IMAGE_ID_TO_SIZE_MAP = {}

IMAGE_SOURCE_MAP = {
  "coco": file_paths.COCO_IMAGES,
  "flicker30k": file_paths.FLICKER30K,
  "visual_news": file_paths.VISUAL_NEWS,
}


def get_image_file(image_id) -> str:
  """Returns the filepath of an image corresponding to an input image id

  To support multiple datasets, we prefix image_ids with "source/"
  """
  source, key = image_id.split("/", 1)
  if source == "open-images-v6":
    import fiftyone
    return join(fiftyone.config.dataset_zoo_dir, image_id)
  if source in IMAGE_SOURCE_MAP:
    return join(IMAGE_SOURCE_MAP[source], key)
  raise ValueError(f"Unknown image id {image_id}")


def get_image_size(image_id):
  """Return (w, h) of the image"""
  if image_id in _IMAGE_ID_TO_SIZE_MAP:
    return _IMAGE_ID_TO_SIZE_MAP[image_id]

  img_file = get_image_file(image_id)
  size = imagesize.get(img_file)

  _IMAGE_ID_TO_SIZE_MAP[image_id] = size
  return size


def get_coco_image_id(subset, image_id):
  """
  Turns image_id dictionary found in GPV data into a single image_id
  """
  return f'coco/{subset}/COCO_{subset}_{str(image_id).zfill(12)}.jpg'


_IMAGE_TO_SUBSETS = None


def get_coco_subset(image_id: int):
  global _IMAGE_TO_SUBSETS
  if _IMAGE_TO_SUBSETS is None:
    _IMAGE_TO_SUBSETS = {}
    for subset in ["train2014", "val2014"]:
      for image_file in listdir(join(file_paths.COCO_IMAGES, subset)):
        image_id = int(image_file.split("_")[-1].split(".")[0])
        _IMAGE_TO_SUBSETS[image_id] = subset
  return _IMAGE_TO_SUBSETS[image_id]


def get_coco_id_from_int_id(image_id: int):
  return get_coco_image_id(get_coco_subset(image_id), image_id)


def get_hdf5_key_for_text(text):
  s = py_utils.consistent_hash(text)
  return "text/" + s[:2] + "/" + s[2:]


def get_hdf5_key_for_image_id(image_id) -> str:
  """Returns the key we would use in HDF5 for the given image_id

  This mapping is a bit convoluted since we are have tried to keep hdf5 feature files
  backward-compatible with newer image_id formats.
  """

  prefix, key = image_id.split("/", 1)
  if prefix == "coco":
    return "coco/" + key.split(".")[0][-2:] + "/" + key
  if prefix == "openimages":
    return 'openimages/' +  key
  else:
    raise NotImplementedError()


def get_image_id_for_hdf5_key(hdf5_key):
  prefix, key = hdf5_key.split("/", 1)
  if prefix == "coco":
    key = key.split("/", 1)[1]
    return 'coco/' + key
  if prefix == "openimages":
    return hdf5_key
  else:
    raise NotImplementedError(hdf5_key)


def enumerate_datasets(group, prefix='', out=None):
  if out is None:
    out = {}

  for k in group:
    if prefix == '':
      full_k = k
    else:
      full_k = prefix + '/' + k
    val = group[k]
    if isinstance(val, h5py.Group):
      enumerate_datasets(val, full_k, out)
    else:
      out[full_k] = val[:]
  return out


def get_clip_image_cache(clip_model):
  src = CLIP_VECTOR_CACHE[clip_model]
  tensor = []
  image_id_to_ix = {}
  with h5py.File(src) as f:
    data = enumerate_datasets(f)

  for i, (k, v) in enumerate(data.items()):
    tensor.append(v)
    image_id_to_ix[get_image_id_for_hdf5_key(k)] = i
  return image_id_to_ix, torch.as_tensor(tensor)


def get_cached_image_vectors(clip_model, image_ids):
  src = CLIP_VECTOR_CACHE[clip_model]
  vectors = []
  with h5py.File(src) as f:
    for image_id in image_ids:
      vectors.append(f[get_hdf5_key_for_image_id(image_id)][:])
  return vectors

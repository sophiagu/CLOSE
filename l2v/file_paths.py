from os import mkdir
from os.path import join, dirname, expanduser, exists

DATA_DIR = expanduser("~/data")

NOCAPS_HOME = join(DATA_DIR, "nocaps")

CC3M = join(DATA_DIR, "cc3m")

COCO_SOURCE = join(DATA_DIR, "coco")
COCO_ANNOTATIONS = join(COCO_SOURCE, "annotations")
COCO_IMAGES = join(COCO_SOURCE, "images")

COCO_SCE_HOME = join(DATA_DIR, "gpv/learning_phase_data/coco_captions/gpv_split")

VQAE = join(DATA_DIR, "vqa-e")

SNLI_VE_HOME = join(DATA_DIR, "SNLI_VE")
FLICKER30K = join(DATA_DIR, "SNLI_VE", "Flickr30K", "flickr30k_images")

VISUAL_NEWS = join(DATA_DIR, "visual_news/origin")

HOME = dirname(dirname(__file__))
CACHE_DIR = join(HOME, "cache")
CLIP_VECTOR_CACHE = {
  "ViT-L/14": join(CACHE_DIR, "coco-images.hdf5")
}
VQA_ANNOTATIONS = join(DATA_DIR, "vqa")

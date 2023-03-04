import argparse
import logging
import os

from transformers import AutoConfig

from l2v.data.visual_entailment import VisualEntailment, EntailmentEvaluator, MNLI
from l2v.experiments.build_trainer import add_train_args, run_trainer
from l2v.experiments.utils import get_adapter
from l2v.model.clip_cap_layers import *
from l2v.model.layers import *
from l2v.train.optimizer import AdamWBuilder, DelayedWarmupScheduleBuilder
from l2v.train.trainer import TrainerSimple
from l2v.utils import py_utils
from l2v.eval.compute_predictions import eval_generative_model


os.environ["TOKENIZERS_PARALLELISM"] = "false"

from l2v.data.coco_captioning import CocoCaptioning
from l2v.model.clip_t5_model import ClipT5Model, LinearAdapter, CLIP_DIMS
from l2v.model.model import BeamSearchSpec
from l2v.train.evaluator import CaptionEvaluator, ResultKey
from l2v.train.trainer_complex import TrainerDataset, EvaluationSetup
from l2v.utils.pytorch_utils import get_devices


def main():
  parser = argparse.ArgumentParser()

  # Model args
  parser.add_argument("--clip_model", default="ViT-L/14")
  parser.add_argument("--t5_model", default="t5-base")

  parser.add_argument("--l_adapter", default="none")
  parser.add_argument("--noise", type=float, default=0.0)
  parser.add_argument("--scale", type=float)

  parser.add_argument("--dropout", type=float, default=0.0)
  parser.add_argument("--train_l", default="always")

  # Optimizer args
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--warmup", type=int, default=None)
  parser.add_argument("--decay", default="linear")

  # Other training args
  parser.add_argument("--batch_size", default=128, type=int)
  parser.add_argument("--epochs", default=8, type=int)

  # Where to save things
  parser.add_argument("--override", action="store_true")
  parser.add_argument("--output_dir")

  parser.add_argument("--debug", action="store_true",
                      help="Train with tiny model/dataset for debugging")

  args = parser.parse_args()

  py_utils.add_stdout_logger()
  l_adapter = get_adapter(args)

  dbg = args.debug

  print("Using T5 Model")
  model = ClipT5Model(args.clip_model, args.t5_model, LinearAdapter(4),
                      caption_l="1to1", language_shift=l_adapter,
                      lowercase_target=True, train_on_l=args.train_l,
                      one_to_many_loss="sum")
  prediction_args = dict(beam_search_spec=BeamSearchSpec(1, 30))

  scheduler = DelayedWarmupScheduleBuilder(warmup=None, decay="linear")

  trainer = TrainerSimple(
    train_dataset=VisualEntailment("train", use_images=True, sample=8 if dbg else None),
    eval_dataset=VisualEntailment("val", use_images=True, sample=10 if dbg else None),
    optimizer=AdamWBuilder(lr=args.lr, weight_decay=0.0, parameter_groups=[]),
    epochs=args.epochs,
    batch_size=args.batch_size,
    save_each_epoch=[8],
    evaluator=EntailmentEvaluator(),
    prediction_args=prediction_args,
    scheduler=scheduler,
    best_model_key=ResultKey("accuracy"),
    num_workers=3
  )

  trainer.train(model, args.output_dir, override=args.override)


if __name__ == '__main__':
  main()
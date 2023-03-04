from l2v.model.language_adapters import *
from l2v.model.layers import Sequential


def get_adapter(args):
  if args.l_adapter.startswith("shift"):
    _, src = args.l_adapter.split("-", maxsplit=1)
    print(f"Using const shift with src={src} scale={args.scale}")
    l_adapter = Shift(src, args.scale, args.noise, renorm=False)
  elif args.l_adapter == "rng1":
    l_adapter = CovNoise("random", args.noise, cov=False, shift=True)
  elif args.l_adapter == "lin":
    l_adapter = LinearAdapter("kp-linear-v1", args.noise, renorm=True)
  elif args.l_adapter == "cc3m-lin":
    l_adapter = LinearAdapter("cc3m-linear-v1", args.noise, renorm=True)
  elif args.l_adapter == "cov":
    l_adapter = CovNoise("coco-cap-kp-restval-av", args.noise)
  elif args.l_adapter == "cc3m-cov":
    l_adapter = CovNoise("cc3ms200k", args.noise)
  elif args.l_adapter == "vis-news-shift":
    l_adapter = CovNoise("kp-restval", args.noise, cov=False)
  elif args.l_adapter == "vis-news-cov":
    l_adapter = CovNoise("kp-restval", args.noise)
  else:
    if args.l_adapter == "none":
      l_adapter = []
    elif args.l_adapter == "bias":
      l_adapter = [CocoCapMeanDiff()]
    else:
      l_adapter = [TrainedAdapter(f"{args.l_adapter}/r0")]

    if not l_adapter:
      l_adapter = AddGuassianNoise(args.noise, renormalize=True)
    else:
      l_adapter = Sequential(
        [AddGuassianNoise(args.noise, renormalize=False)] +
        l_adapter +
        [AddGuassianNoise(0.0, renormalize=True)]
      )
  return l_adapter

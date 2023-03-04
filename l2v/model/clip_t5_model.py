import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Dict, Tuple, Union, Optional

import numpy as np
import clip
import six
import torch
from PIL import Image
from allennlp.common import Registrable, Params
from clip.model import QuickGELU
from torch import nn
from torchvision.transforms import transforms
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoConfig, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Block

from l2v.data.coco_captioning import CaptioningExample
from l2v.data.visual_entailment import VisualEntailmentExample
from l2v.data.visual_news import VisualNewsExample
from l2v.data.vqa_v2 import VqaExample
from l2v.model.layers import Layer
from l2v.model.model import Model, PredictionArg, ExampleOutput
from l2v.train.allennlp_beamsearch import t5_initialize_decoding
from l2v.utils import image_utils, pytorch_utils
from torch.nn import functional as F


CLIP_DIMS = {
  "ViT-B/32": 512,
  "ViT-L/14": 768,
  "RN101": 512,
  "RN50": 1024,
  "RN50x4": 640,
  "RN50x16": 768,
  "RN50x64": 1024
}


class ExampleConverter(Registrable):

  def convert(self, x):
    raise NotImplementedError()

  def convert_train(self, x):
    raise NotImplementedError()

  def get_cls(self):
    raise NotImplementedError()


class ClipT5Adapter(Registrable, nn.Module):

  def init(self, t5_dim, clip_dim):
    pass


@ClipT5Adapter.register("linear-with-shift")
class LinearWithShiftAdapter(ClipT5Adapter):

  def __init__(self, n_tokens: int, n_constant: int=0, dropout=0.0, gnoise=0.0):
    super().__init__()
    self.gnoise = gnoise
    self.n_tokens = n_tokens
    self.dropout = dropout
    self.n_constant = n_constant

  def init(self, t5_dim, clip_dim):
    self.t5_dim = t5_dim
    self.lin = nn.Linear(clip_dim, t5_dim*self.n_tokens)
    if self.n_constant:
      self.constant_tokens = nn.Parameter(torch.zeros(self.n_constant, t5_dim))

  def forward(self, clip_features, is_l):
    if is_l:
      clip_features = clip_features + torch.as_tensor(SHIFT, device=clip_features.device)
      clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
    clip_features = F.dropout(clip_features, self.dropout, self.training)
    if self.training and self.gnoise:
      clip_features = clip_features + torch.randn_like(clip_features)*self.gnoise
    seq = self.lin(clip_features).reshape(-1, self.n_tokens, self.t5_dim)
    if self.n_constant:
      seq = torch.cat([self.constant_tokens.unsqueeze(0).tile(seq.size(0), 1, 1), seq], 1)
    return seq


@ClipT5Adapter.register("debug")
class DebugAdapter(ClipT5Adapter):

  def _to_params(self, quiet=False):
    import inspect
    return dict(code=inspect.getsource(DebugAdapter))

  def init(self, t5_dim, clip_dim):
    self.n_tokens = 4
    self.t5_dim = t5_dim
    self.lin = nn.Linear(clip_dim, t5_dim*self.n_tokens)
    self.ff0 = nn.Linear(clip_dim, t5_dim*self.n_tokens*2)
    self.ff1 = nn.Linear(t5_dim*self.n_tokens*2, t5_dim*self.n_tokens)


  def forward(self, x):
    h = F.dropout(QuickGELU()(self.ff0(x)), 0.1, self.training)
    return (self.ff1(h) + self.lin(x)).view(-1, 4, self.t5_dim)


@ClipT5Adapter.register("linear")
class LinearAdapter(ClipT5Adapter):

  @classmethod
  def from_params(cls, params: Params, *args, **extras):
    if "dropout" or "gnoise" in params:
      if params.get("dropout") or params.get("gnoise"):
        logging.warning("Noise no longer supported!")
      if "dropout":
        del params["dropout"]
      if "gnoise":
        del params["gnoise"]
    return super().from_params(params, *args, **extras)

  def __init__(self, n_tokens: int, n_constant: int=0, dropout=0.0, gnoise=0.0):
    super().__init__()
    self.gnoise = gnoise
    self.n_tokens = n_tokens
    self.dropout = dropout
    self.n_constant = n_constant

  def init(self, t5_dim, clip_dim):
    self.t5_dim = t5_dim
    self.lin = nn.Linear(clip_dim, t5_dim*self.n_tokens)
    if self.n_constant:
      self.constant_tokens = nn.Parameter(torch.zeros(self.n_constant, t5_dim))

  def forward(self, clip_features):
    seq = self.lin(clip_features).reshape(-1, self.n_tokens, self.t5_dim)
    if self.n_constant:
      seq = torch.cat([self.constant_tokens.unsqueeze(0).tile(seq.size(0), 1, 1), seq], 1)
    return seq


@ClipT5Adapter.register("layer-adapter")
class LayerAdapter(ClipT5Adapter):
  def __init__(self, layer: Layer, n_constant=0):
    super().__init__()
    self.layer = layer
    self.n_constant = n_constant

  def init(self, t5_dim, clip_dim):
    self.t5_dim = t5_dim
    if self.n_constant:
      self.constant_tokens = nn.Parameter(torch.zeros(self.n_constant, t5_dim))

  def forward(self, x):
    out = self.layer(x)
    dim = out.size(-1)
    assert (dim % self.t5_dim) == 0
    out = out.view(-1, dim // self.t5_dim, self.t5_dim)
    if self.n_constant:
      out = torch.cat([self.constant_tokens.unsqueeze(0).tile(out.size(0), 1, 1), out], 1)
    return out


@ClipT5Adapter.register("transformer")
class TransformerAdapter(ClipT5Adapter):

  def __init__(self, n_vis_tokens: int, n_constant_tokens: int, num_heads: int):
    super().__init__()
    self.n_vis_tokens = n_vis_tokens
    self.n_constant_tokens = n_constant_tokens
    self.num_heads = num_heads

  def init(self, t5_dim, clip_dim):
    self.t5_dim = t5_dim
    self.lin = nn.Linear(clip_dim, t5_dim*self.n_tokens)
    if self.constant > 0:
      self.constant = nn.Parameter(torch.zeros(self.n_tokens, t5_dim))
    self.transformer = T5Block(T5Config(d_model=t5_dim, num_heads=self.num_heads))

  def forward(self, clip_features):
    seq = self.lin(clip_features).reshape(-1, self.n_tokens, self.t5_dim)
    if self.constant > 0:
      c = self.constant.unsqueeze(0).repeat(seq.size(0), 1, 1)
      seq = torch.cat([c, seq])
    return self.transformer(hidden_states=seq)


@dataclass
class TrainingExample:
  image_id: Optional[str] = None
  """image id if this example has an image input"""

  target_text: Union[List[str], None] = None
  """Texts to generate from this inut"""

  input_text: Optional[str] = None
  """Text input"""

  image_text: Union[str, List[str], None] = None
  """Image description(s) to use instead of a real image"""

  example_id: Optional[str] = None

  def get_example_id(self):
    return self.example_id


@dataclass
class Collate:
  tokenizer: Any
  pre: Any
  encode_image: bool
  image_id_to_ix: Optional[Dict] = None
  clip_model_cache: str = None

  def __call__(self, batch: List[TrainingExample]):
    no_image = []
    has_image = []
    for ex in batch:
      if ex.image_id is not None:
        has_image.append(ex)
      else:
        no_image.append(ex)
    batch = has_image + no_image

    out = {}
    if batch[0].target_text is not None:
      texts = []
      mapping = []
      for batch_ix, x in enumerate(batch):
        texts += x.target_text
        mapping += [batch_ix]*len(x.target_text)
      out["target_mapping"] = torch.as_tensor(mapping, dtype=torch.long)
      labels = self.tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True)
      out["target_ids"] = labels["input_ids"]

    if batch[0].input_text is not None:
      texts = [x.input_text for x in batch]
      labels = self.tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True)
      out["input_ids"] = labels["input_ids"]
      out["input_attention_mask"] = labels["attention_mask"]

    if has_image:
      if self.image_id_to_ix:
        # Assume this are cached in memory, so just get the IDs
        out["clip_images"] = torch.as_tensor(
          [self.image_id_to_ix[x.image_id] for x in has_image], dtype=torch.long)
      elif self.clip_model_cache:
        # Load from disk
        out["clip_images"] = image_utils.get_cached_image_vectors(
          self.clip_model_cache, [x.image_id for x in has_image])
      else:
        # Load the images so we can process them directly
        images = []
        for ex in has_image:
          with Image.open(image_utils.get_image_file(ex.image_id)) as f:
            images.append(self.pre(f))
        out["clip_images"] = torch.stack(images, 0)
    else:
      out["clip_images"] = None

    if no_image:
      if all(isinstance(x.image_text, str) for x in batch):
        out["clip_text"] = clip.tokenize([x.image_text for x in no_image], truncate=True)
      else:
        texts = []
        mapping = []
        for batch_ix, x in enumerate(no_image):
          if isinstance(x.image_text, str):
            texts.append(x.image_text)
            mapping.append(batch_ix)
          else:
            texts += x.image_text
            mapping += [batch_ix]*len(x.image_text)
        out["clip_text"] = clip.tokenize(texts, truncate=True)
        out["text_mapping"] = torch.as_tensor(mapping, dtype=torch.long)
    else:
      out["clip_text"] = None

    return out


@Model.register("clip-t5")
class ClipT5Model(Model):

  @classmethod
  def from_params(
    cls, params: Params, constructor_to_call=None,
    constructor_to_inspect=None, **extras
  ):
    # Backwards compatibility changes
    if "shuffle_cap" in params:
      del params["shuffle_cap"]
    if params["train_on_l"] is False:
      params["train_on_l"] = "never"
    elif params["train_on_l"] is True:
      params["train_on_l"] = "always"
    if "use_image_cache" in params:
      del params["use_image_cache"]
    if "dont_save_clip" not in params:
      params["dont_save_clip"] = False
    if "vqa_targets" in params:
      del params["vqa_targets"]
    if "max_text_len" in params:
      del params["max_text_len"]
    if "random_image" in params:
      del params["random_image"]
    return super().from_params(params, constructor_to_call, constructor_to_inspect, **extras)

  def __init__(self, clip_model: str, t5_model_name: str,
               adapter: ClipT5Adapter,
               language_shift: Layer=None,
               train_on_l: str="never", one_to_many_loss="sum", lowercase_target=False,
               freeze="none", caption_l="other-target", vqa_mode="valid",
               openai_clip=None,
               dont_save_clip=True, image_cache=False, converter: List[ExampleConverter]=None):
    """
    :param t5_model_name: Language model name
    :param adapter: Maps CLIP vectors to tokens to use for the LM
    :param language_shift: Pre-processes the language vectors
    :param one_to_many_loss: How to handle multiple target generation for one set of inputs
    :param lowercase_target: Should we lowercase the target text
    :param train_on_l: When to train on text, can be
                       (always, never, both, optional, skip-lang, skip-image).
    :param caption_l: how to convert captions to text-to-text training example, can be
                      (1to1, other-target, average)
    :param freeze:
    :param vqa_targets:
    :param dont_save_clip:
    :param image_cache:
    """
    super().__init__()
    self.openai_clip = openai_clip
    self.converter = [] if converter is None else converter
    self._converted_map = {x.get_cls(): x for x in self.converter}
    self.language_shift = language_shift
    self.lowercase_target = lowercase_target
    self.clip_model = clip_model
    self.t5_model_name = t5_model_name
    self.adapter = adapter
    self.train_on_l = train_on_l
    self.caption_l = caption_l
    self.one_to_many_loss = one_to_many_loss
    self.freeze = freeze
    self.dont_save_clip = dont_save_clip
    self.image_cache = image_cache
    self.vqa_mode = vqa_mode

    # Set during init
    self._clip_model = None
    self._clip_pre = None
    self._t5_model = None
    self.tokenizer = None
    self.image_id_to_ix = None

    # Prediction args
    self.test_on_l = False
    self.beam_search_spec = None

  def initialize(self, load_params=True):
    if self.openai_clip:
      import open_clip
      logging.info(f"Loading clip {self.clip_model}/{self.openai_clip}...")
      model, _, preprocess = open_clip.create_model_and_transforms(
        self.clip_model, pretrained=self.openai_clip)
    else:
      logging.info(f"Loading clip {self.clip_model}...")
      model, preprocess = clip.load(self.clip_model)
    clip_dim = CLIP_DIMS[self.clip_model]
    self._clip_pre = preprocess
    self._clip_model = model
    for param in self._clip_model.parameters():
      param.requires_grad = False

    logging.info(f"Loading T5 {self.t5_model_name}...")
    if load_params:
      if self.t5_model_name == "t5_base":
        self._t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model_name)
      else:
        self._t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model_name)
    else:
      self._t5_model = T5ForConditionalGeneration(AutoConfig.from_pretrained(self.t5_model_name))
    t5_dim = self._t5_model.encoder.config.d_model


    if self.freeze == "all":
      for param in self._t5_model.parameters():
        param.requires_grad = False
    elif self.freeze != "none":
      raise ValueError()

    if self.image_cache == "memory":
      logging.info("Loading clip vector cache...")
      ixs, tensor = image_utils.get_clip_image_cache(self.clip_model)
      self.register_buffer("_clip_vector_cache", tensor=tensor, persistent=False)
      self.image_id_to_ix = ixs
    else:
      self.image_id_to_ix = None

    self.tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name)

    self.adapter.init(t5_dim, clip_dim)

  def get_collate(self, is_train=False) -> Callable[[List], Dict[str, Any]]:
    if is_train:
      pre = transforms.Compose([
        # transforms.RandomHorizontalFlip(0.2),
        # transforms.RandomApply([transforms.Grayscale(3)], 0.2),
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], 0.2),
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], 0.2),
        # transforms.RandomApply([transforms.RandomPosterize(bits=2)], 0.2)
      ] + self._clip_pre.transforms)
    else:
      pre = self._clip_pre
    if is_train:
      encode_image = not self.train_on_l
    else:
      encode_image = not self.test_on_l
    return Collate(self.tokenizer, pre, encode_image, self.image_id_to_ix,
                   self.clip_model if self.image_cache == "hdf5" else None)

  def preprocess_example_train(self, ex) -> List:
    if ex.__class__ in self._converted_map:
      out = self._converted_map[ex.__class__].convert_train(ex)

    elif isinstance(ex, CaptioningExample):

      extract_i = (self.train_on_l in {"never", "both"} or
        (self.train_on_l in {"optional", "skip-lang"} and ex.image_id is not None))

      if extract_i:
        out = [TrainingExample(ex.image_id, ex.captions)]
      else:
        out = []

      extract_l = (self.train_on_l in {"always", "both"} or
                   (self.train_on_l in {"optional", "skip-image"} and ex.image_id is None))

      if extract_l:
        if self.caption_l == "1to1":
          out += [TrainingExample(None, [x], image_text=x) for x in ex.captions]

        elif self.caption_l == "average":
          out.append(TrainingExample(None, target_text=ex.captions, image_text=ex.captions))

        elif self.caption_l == "other-target":
          targets = ex.captions
          for i, text in enumerate(targets):
            out.append(TrainingExample(
              None, targets[:i] + targets[i+1:], image_text=text))
        else:
          raise NotImplementedError(self.caption_l)

    elif isinstance(ex, VisualEntailmentExample):
      extract_l = (self.train_on_l in {"always", "both"} or
                   (self.train_on_l in {"optional", "skip-image"} and ex.image_id is None))
      out = [TrainingExample(
        None if extract_l else ex.image_id, [ex.label], ex.hypothesis,
        image_text=ex.premise if extract_l else None, example_id=ex.example_id)]

    elif isinstance(ex, VqaExample):
      extract_i = (self.train_on_l in {"never", "both"} or
                   (self.train_on_l in {"optional", "skip-lang"} and ex.image_id is not None))
      target_text = []
      if isinstance(ex.answers, Counter):
        if self.vqa_mode == "valid":
          on = None
          for w, c in ex.answers.most_common():
            if on is None:
              target_text.append(w)
              on = c
            elif c == on or c >= 3:
              target_text.append(w)
            else:
              break
        else:
          raise NotImplementedError()
      elif isinstance(ex.answers, list):
        target_text = ex.answers
      else:
        assert isinstance(ex.answers, str)
        target_text = [ex.answers]
      out = [TrainingExample(
        ex.image_id if extract_i else None, target_text, ex.question,
        None if extract_i else ex.image_text,
        example_id=ex.example_id
      )]

    elif isinstance(ex, VisualNewsExample):
      extract_i = (self.train_on_l in {"never", "both"} or
                   (self.train_on_l in {"optional", "skip-lang"} and ex.image_id is not None))

      out = [
        TrainingExample(
          example_id=ex.example_id,
          image_id=ex.image_id if extract_i else None, 
          input_text=ex.article,
          image_text=ex.caption,
          target_text=[ex.caption])
      ]

    else:
      raise NotImplementedError()

    if self.lowercase_target:
      for ex in out:
        ex.target_text = [x.lower() for x in ex.target_text]

    if self.one_to_many_loss == "individual-sum":
      # Examples with multiple targets are split into an individual example
      flat = []
      for ex in out:
        for target in ex.target_text:
          flat.append(replace(ex, target_text=[target]))
      return flat
    else:
      return out

  def preprocess_example(self, example) -> Any:
    if example.__class__ in self._converted_map:
      return self._converted_map[example.__class__].convert_train(example)

    if isinstance(example, CaptioningExample):
      if example.captions:
        cap = example.captions[np.random.randint(0, len(example.captions))]
      else:
        cap = None
      return TrainingExample(example_id=example.get_example_id(), image_id=example.image_id,
                             target_text=None, input_text=None, image_text=cap)
    elif isinstance(example, VqaExample):
      return TrainingExample(example_id=example.example_id, image_id=example.image_id,
                             target_text=None, input_text=example.question, image_text=None)

    elif isinstance(example, VisualEntailmentExample):
      return TrainingExample(example_id=example.example_id, image_id=example.image_id,
                             input_text=example.hypothesis,
                             target_text=None, image_text=example.premise)

    elif isinstance(example, VisualNewsExample):
      return TrainingExample(example_id=example.example_id,
                             image_id=example.image_id,
                             input_text=example.article,
                             image_text=example.caption,
                             target_text=None)

    else:
      raise NotImplementedError()

  def set_prediction_args(self, beam_search_spec, test_on_l=False, image_cache=None):
    self.beam_search_spec = beam_search_spec
    self.test_on_l = test_on_l
    if image_cache is not None:
      if image_cache == "none":
        self.image_cache = None
        self.image_id_to_ix = None
        self.register_buffer("_clip_vector_cache", None)
      else:
        raise NotImplementedError()

  def _encode(self, clip_images, clip_text, input_ids,
              text_mapping, input_attention_mask):
    clip_features = []
    if clip_images is not None:
      if self.image_cache:
        image_fe = self._clip_vector_cache[clip_images]
      else:
        with torch.no_grad():
          image_fe = self._clip_model.encode_image(clip_images)
      image_fe = image_fe.float()
      image_fe = image_fe / image_fe.norm(dim=-1, keepdim=True)
      clip_features.append(image_fe)

    if clip_text is not None:
      with torch.no_grad():
        text_fe = self._clip_model.encode_text(clip_text)
      text_fe = text_fe.float()
      if text_mapping is not None:
        text_fe = pytorch_utils.segment_mean(text_fe, text_mapping)
      text_fe = text_fe / text_fe.norm(dim=-1, keepdim=True)
      text_fe = self.language_shift(text_fe)
      clip_features.append(text_fe)

    clip_features = torch.cat(clip_features, 0)
    clip_tokens = self.adapter(clip_features)

    # assert clip_images is not None
    #
    if input_ids is not None:
      input_embed = self._t5_model.shared(input_ids)
      input_embed, input_mask = pytorch_utils.concat_masked_sequences(
        clip_tokens, None,
        input_embed, input_attention_mask
      )
    else:
      input_embed = clip_tokens
      input_mask = None
    encoding = self._t5_model.encoder(
      inputs_embeds=input_embed,
      return_dict=True
    ).last_hidden_state
    return encoding, input_mask

  def forward(self, clip_images, clip_text, target_ids, target_mapping=None,
              text_mapping=None,
              input_ids=None, input_attention_mask=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_ids = target_ids.masked_fill(
      target_ids == self.tokenizer.pad_token_id, -100)
    encoder_out, input_mask = self._encode(clip_images, clip_text, input_ids,
                                           text_mapping, input_attention_mask)

    if target_mapping is not None:
      encoder_out = encoder_out[target_mapping]
      if input_mask is not None:
        input_mask = input_mask[target_mapping]

    out: Seq2SeqLMOutput = self._t5_model(
      encoder_outputs=(encoder_out, ),
      attention_mask=input_mask,
      labels=target_ids,
      return_dict=True
    )

    if self.one_to_many_loss in {"sum", "individual-sum"}:
      loss = out.loss
    elif self.one_to_many_loss in {"logsumexp", "joint"}:
      batch, seq, dim = out.logits.size()
      loss = F.cross_entropy(out.logits.view(-1, dim), target_ids.view(-1), reduction="none")
      loss = loss.view(batch, seq).sum(-1)
      loss = loss / (target_mapping >= 0).float().sum(-1)
      losses = []
      on = 0
      for c in torch.unique_consecutive(target_mapping, return_counts=True)[1]:
        losses.append(torch.logsumexp(loss[on:on+c], 0))
        on += c
      loss = torch.stack(losses).mean()
      if self.one_to_many_loss == "joint":
        loss = (loss + out.loss) / 2.0
    else:
      raise NotImplementedError()
    return loss, {}

  def predict(self, clip_images, clip_text, target_ids=None, text_mapping=None,
              input_ids=None, input_attention_mask=None):
    # Use no_grad just so clients don't have to remember to
    enc, input_mask = self._encode(clip_images, clip_text, input_ids,
                                   text_mapping, input_attention_mask)

    bs = self.beam_search_spec.build(self.tokenizer.eos_token_id)
    decode_init = t5_initialize_decoding(
      self.tokenizer, self._t5_model, enc, input_mask)
    input_ids, logprobs = bs.search(*decode_init)

    logprobs = logprobs.cpu().numpy()
    input_ids = input_ids.cpu().numpy()

    out_text = []
    for batch in range(len(input_ids)):
      text = [self.post_process_generation(x) for x in input_ids[batch]]
      out_text.append(text)

    return [ExampleOutput(txt, p.tolist()) for txt, p in zip(out_text, logprobs)]

  def post_process_generation(self, generated_ids):
    return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

  # Override these methods to skip saving clip, note this will
  # not work if this is a sub-module of the thing we are getting a
  # state dict for
  # TODO can we fix this by simply not registering the clip model?

  def state_dict(self, *args, **kwargs):
    if self.dont_save_clip:
      tmp = self._clip_model
      self._clip_model = None
      out = super().state_dict(*args, **kwargs)
      self._clip_model = tmp
      return out
    else:
      return super().state_dict(*args, **kwargs)

  def load_state_dict(self, state_dict, strict: bool = True):
    if self.dont_save_clip:
      tmp = self._clip_model
      self._clip_model = None
      out = super().load_state_dict(state_dict, strict)
      self._clip_model = tmp
      return out
    else:
      super().load_state_dict(state_dict, strict)







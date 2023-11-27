from transformers import PreTrainedTokenizer
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class tokenizer_ctx(ContextDecorator):
  tokenizer: PreTrainedTokenizer
  truncation_side: Optional[str]
  padding_side: Optional[str]
  orig_truncation_side: Optional[str] = field(default=None, init=False)
  orig_padding_side: Optional[str] = field(default=None, init=False)

  def __enter__(self):
    self.orig_truncation_side = self.tokenizer.truncation_side
    self.orig_padding_side = self.tokenizer.padding_side
    if self.truncation_side is not None:
      self.tokenizer.truncation_side = self.truncation_side
    if self.padding_side is not None:
      self.tokenizer.padding_side = self.padding_side
    return self

  def __exit__(self, *exc):
    self.tokenizer.truncation_side = self.orig_truncation_side
    self.tokenizer.padding_side = self.orig_padding_side
    return False
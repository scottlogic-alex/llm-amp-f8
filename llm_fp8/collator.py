from torch import LongTensor, tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, BatchEncoding
from typing import Sequence, List, Optional
from contextlib import nullcontext
import copy

from .collator_types import DataInstance, CollatedData
from .tokenizer_ctx import tokenizer_ctx
from .causal_gpt import IGNORE_INDEX

@dataclass
class DataCollatorForCausalLM(object):
  tokenizer: PreTrainedTokenizerBase
  source_max_len: int
  target_max_len: int
  train_on_source: bool
  predict_with_generate: bool
  truncate_toward_center: bool
  use_bos_token_in_prompt: bool
  simulate_worst_case_seq_len: bool
  output_mask: bool

  def __call__(self, instances: Sequence[DataInstance]) -> CollatedData:
    # Extract elements
    sources: List[str] = [f"{self.tokenizer.bos_token if self.use_bos_token_in_prompt else ''}{example['input']}" for example in instances]
    targets: List[str] = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
    # Tokenize
    with tokenizer_ctx(self.tokenizer, truncation_side='left') if self.truncate_toward_center else nullcontext():
      tokenized_sources_with_prompt: BatchEncoding = self.tokenizer(
        sources,
        max_length=self.source_max_len,
        truncation=True,
        add_special_tokens=False,
      )
    with tokenizer_ctx(self.tokenizer, truncation_side='right') if self.truncate_toward_center else nullcontext():
      tokenized_targets: BatchEncoding = self.tokenizer(
        targets,
        max_length=self.target_max_len,
        truncation=True,
        add_special_tokens=False,
      )
    # Build the input and labels for causal LM
    input_ids: List[LongTensor] = []
    labels: List[LongTensor] = []
    for tokenized_source, tokenized_target in zip(
      tokenized_sources_with_prompt['input_ids'],
      tokenized_targets['input_ids']
    ):
      if not self.predict_with_generate:
        prompt_and_continuation: LongTensor = tensor(tokenized_source + tokenized_target)
        # simulate worst-case sequence length
        if self.simulate_worst_case_seq_len:
          prompt_and_continuation = pad(
            prompt_and_continuation,
            (0, (self.source_max_len + self.target_max_len) - (len(tokenized_source) + len(tokenized_target))),
            mode='constant',
            value=self.tokenizer.pad_token_id,
          )
        input_ids.append(prompt_and_continuation)
        if not self.train_on_source:
          source_ignored: LongTensor = tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
          # simulate worst-case sequence length
          if self.simulate_worst_case_seq_len:
            source_ignored = pad(
              source_ignored,
              (0, (self.source_max_len + self.target_max_len) - (len(tokenized_source) + len(tokenized_target))),
              mode='constant',
              value=self.tokenizer.pad_token_id,
            )
          labels.append(source_ignored)
        else:
          labels.append(tensor(copy.deepcopy(tokenized_source + tokenized_target)))
      else:
        input_ids.append(tensor(tokenized_source))
    # Apply padding
    input_ids: LongTensor = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    labels: Optional[LongTensor] = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
    data_dict: CollatedData = { 'input_ids': input_ids }
    if self.output_mask:
      # Dao flash attn doesn't support masking, and instead relies on masking the loss via IGNORE_INDEX tokens
      data_dict['attention_mask'] = input_ids.ne(self.tokenizer.pad_token_id)
    if labels is not None:
      data_dict['labels'] = labels
    return data_dict
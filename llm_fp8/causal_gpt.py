from torch import LongTensor, FloatTensor
from typing import Optional, Union, NamedTuple
from flash_attn.models.gpt import GPTLMHeadModel
from torch.nn import CrossEntropyLoss
from transformers import GPT2Config
from transformers.utils import logging

class CausalLMOutput(NamedTuple):
  logits: FloatTensor

class LossAndLogits(NamedTuple):
  loss: FloatTensor
  logits: FloatTensor

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100

class CausalGPTLMHeadModel(GPTLMHeadModel):
  loss_fn: CrossEntropyLoss
  def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
    super().__init__(
      config,
      process_group=process_group,
      device=device,
      dtype=dtype,
    )
    self.loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

  # this doesn't have everything you'd want for inference (like a kv cache, or attention masking)
  # and it only supports padded data (via loss masking), not packed.
  def forward(
    self,
    input_ids: Optional[LongTensor] = None,
    attention_mask: Optional[FloatTensor] = None,
    position_ids: Optional[LongTensor] = None,
    labels: Optional[LongTensor] = None,
  ) -> Union[CausalLMOutput, LossAndLogits]:
    r"""
    labels (`LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
      Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
      `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
      ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
    ```"""
    assert attention_mask is None, "Dao flash attn doesn't accept attn mask"

    outputs: CausalLMOutput = super().forward(
      input_ids,
      position_ids=position_ids,
    )
    if labels is None:
      return outputs

    logits: FloatTensor = outputs.logits
    # we are doing next-token prediction; shift prediction scores and input ids by one
    shift_logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss: FloatTensor = self.loss_fn(shift_logits.flatten(end_dim=-2), labels.flatten())

    return LossAndLogits(
      loss=loss,
      logits=logits,
    )
# import torch
# from torch.optim import AdamW
from transformers import GPTNeoXTokenizerFast
from flash_attn.models.gpt import GPTLMHeadModel, GPT2Config
from flash_attn.models.gpt_neox import GPTNeoXConfig, gpt_neox_config_to_gpt2_config
from dataclasses import dataclass, field
from typing import Optional, TypedDict, Dict
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.formatting.formatting import LazyRow
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from llm_fp8.collator import DataCollatorForCausalLM
from llm_fp8.causal_gpt import CausalGPTLMHeadModel

cache_dir: Optional[str] = None

model_name = 'EleutherAI/pythia-70m-deduped'
gpt_neox_config = GPTNeoXConfig.from_pretrained(model_name, cache_dir=cache_dir)
gpt2_config: GPT2Config = gpt_neox_config_to_gpt2_config(gpt_neox_config)
model: CausalGPTLMHeadModel = CausalGPTLMHeadModel.from_pretrained(model_name, gpt2_config)

DEFAULT_PAD_TOKEN = '[PAD]'
tokenizer: GPTNeoXTokenizerFast = GPTNeoXTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
special_tokens: Dict[str, str] = {}
if tokenizer._pad_token is None:
  special_tokens['pad_token'] = DEFAULT_PAD_TOKEN
tokenizer.add_special_tokens(special_tokens)
# we don't resize the embedding because pad tokens will be masked out anyway

ALPACA_PROMPT_DICT = {
  "prompt_input": (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
  ),
  "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
  ),
}

class Datum(TypedDict):
  input: str

def extract_alpaca_dataset(example: LazyRow) -> Datum:
  prompt_format = ALPACA_PROMPT_DICT['prompt_input' if example.get('input', '') else 'prompt_no_input']
  return Datum(input=prompt_format.format(**example))

dataset: DatasetDict = load_dataset('yahma/alpaca-cleaned')
dataset: DatasetDict = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
dataset: DatasetDict = dataset.remove_columns(
  [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
)
dataset: Dataset = dataset['train']

# alpaca-cleaned doesn't include a test split
split_dataset: DatasetDict = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset: Dataset = split_dataset['train']
test_dataset: Dataset = split_dataset['test']

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
  train_on_source: Optional[bool] = field(
    default=False,
    metadata={"help": "Whether to train on the input in addition to the target text."}
  )
  source_max_len: int = field(
    default=1024,
    metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
  )
  target_max_len: int = field(
    default=256,
    metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
  )
  truncate_toward_center: Optional[bool] = field(
    default=False,
    metadata={"help": "Truncate prompt from left side, truncate continuation from right side."}
  )
  use_bos_token_in_prompt: Optional[bool] = field(
    default=False,
    metadata={"help": "If your model was pretrained to utilise BOS (e.g. LLaMA), then make use of it in prompt."}
  )
  simulate_worst_case_seq_len: bool = field(
    default=False,
    metadata={"help": "pad prompts to maximum size, to help you measure the worst-case memory usage you'll experience in your dataset."}
  )
  group_by_length: bool = field(default=False, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
  remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})

train_args = TrainingArguments(
  output_dir='out',
)

data_collator = DataCollatorForCausalLM(
  tokenizer=tokenizer,
  source_max_len=train_args.source_max_len,
  target_max_len=train_args.target_max_len,
  train_on_source=train_args.train_on_source,
  predict_with_generate=train_args.predict_with_generate,
  truncate_toward_center=train_args.truncate_toward_center,
  use_bos_token_in_prompt=train_args.use_bos_token_in_prompt,
  simulate_worst_case_seq_len=train_args.simulate_worst_case_seq_len,
  output_mask=False,
)

trainer = Seq2SeqTrainer(
  model=model,
  tokenizer=tokenizer,
  data_collator=data_collator,
  args=train_args,
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
)
trainer.train()
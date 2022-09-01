#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import random
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import torch
import datasets
from datasets import load_dataset

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    DataCollatorWithPadding
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    use_fim: bool = field(
        default=False,
        metadata={
            "help": (
                "Implement FIM in-context learning for chunking and applying causal masking to each chunk"
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_path: Optional[str] = field(default=None, metadata={"help": "The input training / validation path (json file)."})
    data_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the cache data"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )


# load github pretraining dataset
def load_data(data_args):
    sub_dirs = ['c', 'cpp', 'java', 'py', 'js', 'rb', 'ts', 'others', 'csharp', 'php']
    train_files, test_files = [], []

    for sub_dir in sub_dirs:
        for split in ['train', 'test']:
            for partition in range(20):
                partition_id = 'partition_id=' + str(partition) 
                local_dataset_path = os.path.join(data_args.data_path, sub_dir, split, partition_id)
                
                file_names = sorted(os.listdir(local_dataset_path))
                file_names = [f for f in file_names if f.endswith(".json")]
                data_files = [local_dataset_path + '/' + f for f in file_names]
            
                if split == "train":
                    train_files.extend(data_files)
                elif split == "test":
                    test_files.extend(data_files)
                else:
                    raise Exception('Not found the corresponding split')

    print(f"Number of train data files: {len(train_files)}")
    print(f"First train file : {train_files[0]}")
    print(f"Last train file : {train_files[-1]}")

    print(f"Number of test data files: {len(test_files)}")
    print(f"First test file : {test_files[0]}")
    print(f"Last test file : {test_files[-1]}")

    if data_args.data_cache:
        # load the whole dataset into a cache directory (not recommended)
       dataset = load_dataset('json', data_files={'train': train_files, 'validation': test_files}, cache_dir=data_args.data_cache)     
    else:
        # instead of loading the whole dataset, stream each element of the dataset
       dataset = load_dataset('json', data_files={'train': train_files, 'validation': test_files}, streaming=True)
    return dataset

# Causal mask language modelling
def get_sentinel(sentinel_tokens, i):
    return sentinel_tokens[i]

def sentinel_masking(document: torch.Tensor, spans: List[Tuple[int, int]], sentinel_tokens: List[int]):
    document_clone = document.clone()
    document_retrieve_mask = torch.ones_like(document_clone).to(torch.bool)

    for i, span in enumerate(spans):
        document_clone[span[0]] = get_sentinel(sentinel_tokens, i)
        document_retrieve_mask[span[0] + 1:span[1]] = False

    return document_clone[document_retrieve_mask]

def sentinel_targets(document: torch.Tensor, spans: List[Tuple[int, int]], sentinel_tokens: List[int], sentinel_end_token: int):
    num_focused_tokens = sum(x[1] - x[0] for x in spans)
    num_spans = len(spans)
    target = torch.zeros(num_focused_tokens + 2 * num_spans).to(document)
    index = 0
    
    assert len(sentinel_tokens) > len(spans)

    for i, span in enumerate(spans):
        target[index] = get_sentinel(sentinel_tokens, i)
        index += 1
        size = span[1] - span[0]
        target[index: index + size] = document[span[0]:span[1]]
        target[index + size] = sentinel_end_token
        index = index + size + 1
    return target

def get_spans_to_mask(document_length: int, sentinel_tokens: List[int]) -> List[Tuple[int, int]]:
    # Ok, we do not use a budget here but instead
    # our goal is to sample from ~ U[0,1] in the case of len(sentinel_tokens) = 1
    # If len(sentinel_tokens) > 1 we try to find len(sentinel_tokens) non intersecting spans
    len_sentinel_tokens = torch.poisson(
        torch.tensor([float(1)])).clamp(1, len(sentinel_tokens) - 1).to(torch.int).item()

    if len_sentinel_tokens == 1:
        start, end = np.random.uniform(size=2)
        if end < start:
            start, end = end, start
        # round down
        start = int(start * document_length)
        # round up
        end = int(end * document_length + 0.5)
        if start == end:
            return None
        else:
            assert start < end
            return [(start, end)]

    # Let's implement the general case. We will create len(self.sentinel_tokens) ** 2 possible candidates
    # And we will filter one by one to insure no intersections. If we can't find anything then so be it.
    return_spans: List[Tuple[int, int]] = []
    candidate_spans: List[Tuple[int, int]] = [
        tuple(np.random.uniform(size=2)) for _ in range(len_sentinel_tokens ** 2)]
    candidate_spans = [(int(start * document_length), int(end *
                        document_length + 0.5)) for (start, end) in candidate_spans]
    candidate_spans = [(start, end) if start <= end else (
        end, start) for (start, end) in candidate_spans]
    while len(return_spans) < len_sentinel_tokens and len(candidate_spans) > 0:
        candidate_span = candidate_spans.pop()
        if not any(span_intersection(x, candidate_span) for x in return_spans):
            return_spans.append(candidate_span)
    return return_spans

def get_ordered_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return sorted(spans, key=lambda x: x[0])


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    raw_datasets = load_data(data_args)


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained mod el and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Setup tokenizer
    pad_token = "<pad>"
    bos_token = "<|endoftext|>"
    bos_token_id = tokenizer.convert_tokens_to_ids([bos_token])[0]

    tokenizer.pad_token = pad_token
    tokenizer.padding_side = "left"

    # Load model
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "file_contents" if "file_contents" in column_names else column_names[0]
    file_column_name = "filename" if "filename" in column_names else column_names[1]

    # preprocessing function
    def preprocess_function(examples):
        # find file extension
        ind = examples[file_column_name].rfind('.')
        file_ext = examples[ind:]

        # append attribute to the training data 
        # for attribute generation and prediction
        attribute = '<| file ext=' + file_ext + ' |>'

        # 50% at the beginning, 50% at the end
        if random.randint(0, 1):
            text = attribute + '\n' + examples[text_column_name]
        else:
            text = examples[text_column_name] + '\n' + attribute

        # tokenization
        output = tokenizer(text)
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )


    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Setup causal mask language modelling
    sentinel_tokens = []
    for i in range(256):
        sentinel_tokens.append(tokenizer.convert_tokens_to_ids([f"<|mask:{i}|>"])[0])
    sentinel_end_token = tokenizer.convert_tokens_to_ids(["<|endofmask|>"])[0]

    # weights for cross entropy loss
    criterion_weights = torch.ones(len(tokenizer))
    for i in range(len(sentinel_tokens)):
        criterion_weights[sentinel_tokens[i]] = 0.0

    # Main data processing function that will generate chunk of block_size.
    
    # Let A_1 A_2 A_3 be beginning, middle, and end of a doc A, and similarly for B_1, B_2, B_3.  Denote chunk boundaries with |||. 
    # Don't concatenate, but chunk and then transform:
    # A_1 A_3 A_2 ||| B_1 B_3 B_2 ||| C_1 C_3 C_2
    # It leads to having lots of padding within the chunk [particularly for docs where (doc_len % 2048) is much less than 2048]
    
    # For docs that were longer than the context window,transformation is applied to each chunk. E.g. if A is a long document with parts A_1 ... A_6:
    # A_1 A_3 A_2 ||| A_4 A_6 A_5 ||| B_1 ...
    # So each chunk contains at most one document, but a document can occur in multiple chunks.

    def chunking_no_fim_function(examples):
        example_length = len(examples[list(examples.keys())[0]])
        total_length = example_length
        # We add padding if the documents length is longer than multiple of block size
        if total_length >= block_size:
            total_length = math.ceil(total_length // block_size) * block_size

        result = {}
        # Split by chunks of block_size.
        for k, t in examples.items():
            for i in range(0, total_length, block_size):
                # The last chunk that does not fit
                context_length = example_length - i
                if block_size > context_length:
                    item = t[i : i + context_length]
                else:
                    item = t[i : i + block_size]
                
                if i == 0:
                    result[k] = []
                    # Exclude bos token from causal masking
                    item = torch.tensor(item[1:])
                    bos_tensor = torch.tensor([bos_token_id])
                
                assert list(item.shape)[0] > 0
                spans = get_spans_to_mask(list(item.shape)[0], sentinel_tokens)
                if spans is None:
                    if i == 0:
                        result[k].append(torch.cat([bos_tensor, item])[:block_size])
                    else:
                        result[k].append(item[:block_size])
                else:
                    spans = get_ordered_spans(spans)
                    causal_source = sentinel_masking(item, spans, sentinel_tokens)
                    causal_masked = sentinel_targets(item, spans, sentinel_tokens, sentinel_end_token)
                    # Save the causal masked input
                    if i == 0:
                        result[k].append(torch.cat([bos_tensor, causal_source, causal_masked][:block_size]))
                    else:
                        result[k].append(torch.cat([causal_source, causal_masked])[:block_size])
        result["labels"] = result["input_ids"].copy()
        return result

    # Main data processing function that will generate chunk of block_size.
    # Concatenate, chunk, split based on <|endoftext|> delimiter and apply transformation on each chunk's document independently, then rejoin
    # A_1 A_3 A_2 B_1 B_3 B_2 C_1 ||| C_3 C_2

    def chunking_fim_function(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        concatenated_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = concatenated_length
        # We add padding if the documents length is longer than multiple of block size
        if total_length >= block_size:
            total_length = math.ceil(total_length // block_size) * block_size
        
        result = {}

        # Split by chunks of block_size.
        for k, t in concatenated_examples.items():
            for i in range(0, total_length, block_size):
                # The last chunk that does not fit
                context_length = concatenated_length - i
                if block_size > context_length:
                    item = t[i : i + context_length]
                else:
                    item = t[i : i + block_size]
                # Within each chunks, split based on bos_token
                item_split = [list(y) for x, y in itertools.groupby(item, lambda z: z == bos_token_id) if not x]

                sentinel_masked_result = []

                # Apply causal masking on each documents within each chunk
                for item in item_split:
                    assert len(item) > 0
                    spans = get_spans_to_mask(len(item), sentinel_tokens)
                    # Convert item to tensor
                    item = torch.tensor(item)
                    bos_tensor = torch.tensor([bos_token_id])

                    if spans is None:
                        sentinel_masked_result.append(torch.cat([bos_tensor, item]))
                    else:
                        spans = get_ordered_spans(spans)
                        causal_source = sentinel_masking(item, spans, sentinel_tokens)
                        causal_masked = sentinel_targets(item, spans, sentinel_tokens, sentinel_end_token)
                        sentinel_masked_result.append(torch.cat([bos_tensor, causal_source, causal_masked]))
                # Rejoin all the documents
                result[k].append(torch.cat(sentinel_masked_result)[:block_size])
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="chunking text then apply causal masking"):
        if model_args.use_fim:
            lm_datasets = tokenized_datasets.map(
                chunking_fim_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                chunking_no_fim_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # padding data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding='max_length', max_length=block_size, return_tensors='pt')

    # define a custom trainer here for weighted cross entropy
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (weighted cross entropy loss)
            loss_fct = nn.CrossEntropyLoss(weight=criterion_weights)
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()

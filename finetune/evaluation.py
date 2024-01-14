#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import datasets
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy
import json

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from .utils.template import encode_with_prompt_completion_format, encode_with_messages_format
from .eval.utils import encode_with_prompt_completion_format_eval, get_next_word_predictions, eval_nli_task, score_completions, score_qa_task

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    # Train arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval_file", type=str, default=None, help="A csv or a json file containing the evaluation data."
    )
    parser.add_argument(
        '--eval_dataset_name',
        type=str,
        default=None,
        help='The name of the dataset to use (via the datasets library).',
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=2,
        help='Batch size for evaluation.',
    )
    parser.add_argument(
        '--eval_task',
        type=str,
        default=None,
        help='Task to evaluate on. Currently only supports "nli".',
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )

    # Tokenizer and batch arguments
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    
    parser.add_argument(
        '--use_lora',
        type=str,
        default=None,
        help='If passed, will load Lora model and use it to train the model.',
    )
    # Save and logging arguments
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # assert when args.dataset_name and args.eval_file are both None
    if args.dataset_name is None and args.eval_file is None:
        raise ValueError("Need either a dataset name or a evaluation file.")
    
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    # raw_datasets['train'] = raw_datasets['train'].shard(1000, 1)
    if args.eval_file is not None:
        eval_raw_dataset = load_dataset(
            "json",
            data_files={"test": args.eval_file},
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path == "meta-llama/Llama-2-7b-hf":
        # Add padding token to the tokenizer
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
        
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            use_flash_attention_2=True if args.use_flash_attn else False,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
        )
        # model = model.to("cuda")
    else:
        print("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    if "input" in raw_datasets["train"].column_names and "output" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'input'&'output' or 'messages' in your column names.")
    
    # Specifically for P3 dataset
    if args.dataset_name is not None:
        eval_dataset = raw_datasets['test']
        # Filtering if only want to specific dataset
        if args.eval_dataset_name is not None:
            eval_dataset = eval_dataset.filter(lambda example: example['dataset'] == args.eval_dataset_name)

    if args.use_lora is not None:
        from peft import PeftModel
        print("resume from checkpoint" + args.use_lora)
        model = PeftModel.from_pretrained(model, args.use_lora, is_trainable=False)
        
    model.eval()
    # P3 specific
    eval_stat = {}
    unique_tasks = list(set(eval_dataset['dataset']))
    print(f"Unique tasks: {unique_tasks}")
    print(f"Number of unique tasks: {len(unique_tasks)}")
    
    for task in tqdm(unique_tasks, desc="Evaluating tasks"):
        print(f"Task: {task}")
        task_dataset = eval_dataset.filter(lambda example: example['dataset'] == task)
        print(f"Number of examples: {len(task_dataset)}")
        eval_acc = score_qa_task(model, tokenizer, task_dataset, args.eval_batch_size)
        print(f"Eval accuracy: {eval_acc}")
        eval_stat[task] = {
            'acc': eval_acc,
            'num_examples': len(task_dataset),
        }
        # calculate average accuracy and weighted average accuracy
    avg_acc = 0
    weighted_avg_acc = 0
    num_examples = 0
    for task in eval_stat:
        weighted_avg_acc += eval_stat[task]['acc'] * eval_stat[task]['num_examples']
        num_examples += eval_stat[task]['num_examples']
    avg_acc = sum([eval_stat[task]['acc'] for task in eval_stat]) / len(eval_stat)
    weighted_avg_acc /= num_examples
    print(f"Average accuracy: {avg_acc}")
    print(f"Weighted average accuracy: {weighted_avg_acc}")

    eval_stat['avg_acc'] = avg_acc
    eval_stat['weighted_avg_acc'] = weighted_avg_acc

    with open(os.path.join(args.output_dir, "eval_stat.json"), 'w') as f:
        json.dump(eval_stat, f)

if __name__ == "__main__":
    main()
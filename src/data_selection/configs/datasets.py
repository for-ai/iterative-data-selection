# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class rte_dataset:
    dataset: tuple = ("bigscience/P3", "super_glue_rte_does_it_follow_that")
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class cb_dataset:
    dataset: tuple = ("bigscience/P3", "super_glue_cb_does_it_follow_that")
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class anli_r1_dataset:
    dataset: str = ("bigscience/P3", "anli_does_it_follow_that_r1")
    train_split: str = "train_r1"
    test_split: str = "test"

@dataclass
class anli_r2_dataset:
    dataset: str = ("bigscience/P3", "anli_does_it_follow_that_r2")
    train_split: str = "train_r2"
    test_split: str = "test"

@dataclass
class anli_r3_dataset:
    dataset: str = ("bigscience/P3", "anli_does_it_follow_that_r3")
    train_split: str = "train_r3"
    test_split: str = "test"

@dataclass
class copa_dataset:
    dataset: tuple = ("bigscience/P3", "super_glue_copa_cause_effect")
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class hellaswag_dataset:
    dataset: tuple = ("bigscience/P3", "hellaswag_complete_first_then")
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class winogrande_dataset:
    dataset: tuple = ("bigscience/P3", "winogrande_winogrande_xl_fill_in_the_blank")
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class wsc_dataset:
    dataset: tuple = ("bigscience/P3", "super_glue_wsc.fixed_replaced_with")
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class wic_dataset:
    dataset: tuple = ("bigscience/P3", "super_glue_wic_question_context")
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class storycloze_dataset:
    dataset: tuple = ("bigscience/P3", "storycloze_complete_first_then")
    train_split: str = "train"
    test_split: str = "validation"
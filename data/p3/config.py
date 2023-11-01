from dataclasses import dataclass

@dataclass
class rte_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "super_glue_rte_does_it_follow_that"
    type: str = "nli"

@dataclass
class cb_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "super_glue_cb_does_it_follow_that"
    type: str = "nli"

@dataclass
class anli_r1_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "anli_does_it_follow_that_r1"
    type: str = "nli"

@dataclass
class anli_r2_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "anli_does_it_follow_that_r2"
    type: str = "nli"

@dataclass
class anli_r3_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "anli_does_it_follow_that_r3"
    type: str = "nli"

@dataclass
class copa_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "super_glue_copa_cause_effect"
    type: str = "sc"

@dataclass
class hellaswag_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "hellaswag_complete_first_then"
    type: str = "sc"

@dataclass
class storycloze_dataset:
    dataset_path: str = "storycloze/"
    prompt_source: str = "story_cloze/2016"
    prompt_template: str = "Choose Story Ending"
    type: str = "sc"

@dataclass
class winogrande_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "winogrande_winogrande_xl_fill_in_the_blank"
    type: str = "wsd"

@dataclass
class wsc_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "super_glue_wsc.fixed_replaced_with"
    type: str = "wsd"

@dataclass
class wic_dataset:
    dataset: str = "bigscience/P3"
    prompt_template: str = "super_glue_wic_question_context"
    type: str = "cr"

DATASETS = {
    "rte": rte_dataset,
    "cb": cb_dataset,
    "anli_r1": anli_r1_dataset,
    "anli_r2": anli_r2_dataset,
    "anli_r3": anli_r3_dataset,
    "copa": copa_dataset,
    "hellaswag": hellaswag_dataset,
    "storycloze": storycloze_dataset,
    "winogrande": winogrande_dataset,
    "wsc": wsc_dataset,
    "wic": wic_dataset,
}
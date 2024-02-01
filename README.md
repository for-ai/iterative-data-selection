# data-selection

## Datasets

### Dataset Specifications
P3 Subset 

| Dataset Name                                                                                                        | Category     |
|---------------------------------------------------------------------------------------------------------------------|-------------|
| rte                                    | `NLI` |
| cb                                    | `NLI` |
| anli_r1                                  | `NLI`  |
| anli_r2                                  | `NLI`  |
| anli_r3 | `NLI` |
| copa | `SC` |
| hellaswag                                        | `SC`  |
| storycloze                                        | `SC`  |
| winogrande                                        | `WSD`  |
| wsc                                        | `WSD`  |
| wic                                        | `CR`  |

`NLI`: Natural Language Inference

`SC`: Sentence Complement

`WSD`: Word Sense Disambiguation

`CR`: Coreference Resolution

### Usage

```python
from datasets import load_dataset
dataset = load_dataset('simonycl/p3_0.5_dataset')
```
The dataset is accessible in [simonycl/p3_0.5_dataset](https://huggingface.co/datasets/simonycl/p3_0.5_dataset) on HuggingFace Datasets Hub.

## Finetuning

### Scripts
```bash
# Galactica-1.3b (Following the 0.5% data settings)
bash scripts/finetune_galactica_with_accelerate.sh

# LLaMA-2-7B 
bash scripts/finetune_llama_with_accelerate.sh
```

## Coreset Selection
Modify `selection/config.yaml` to change the dataset and coreset method settings.
```bash
cd selection
python main.py
```
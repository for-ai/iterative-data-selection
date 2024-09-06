# Selecting Diverse Instructions

This repository contains the official code for the paper: [Efficient Instruction Data Selection
via Clustering and Iterative Refinement](https://arxiv.org/abs/2405.10331).

![KMQ Visualization](visual/kmq.jpg)


## Dataset

To download the datasets used in this project, run [this script](https://github.com/allenai/open-instruct/blob/main/scripts/data/prepare_train_data.sh). We used Alpaca, ShareGPT and WizardLM datasets for training and evaluation.

After downloading, datasets will be stored in the `data/processed` directory of the project.



## Finetuning


## Coreset Selection
The hyperparameters and configurations are managed by [Hydra](https://hydra.cc/). The configurations are stored in `selection/config/`.
You should run the code by executing `main.py` in the `selection` directory. You can also specify the hyperparameters by command line arguments.
```bash
cd selection
python main.py data=[sharegpt|wizardlm] encoder=miniLM coreset=random
```
The selected indices are stored under `selection/indices/`.


### Finetuning
```bash
# Llama-2-7b-hf (with accelerate and deepspeed)
bash scripts/finetune_llama_with_accelerate.sh [INDICES]
```

### Evaluation
```bash
bash scripts/eval/{eval}.sh
```

### Reference
This code is based on the following repository:
- [open-instruct](https://github.com/allenai/open-instruct)


### Citation
If you find this code useful, please cite our paper:
```
@article{
  author = {...},
  title = {Efficient Instruction Data Selection via Clustering and Iterative Refinement},
  journal = {arXiv preprint arXiv:2405.10331},
  year = {2024}
}
```

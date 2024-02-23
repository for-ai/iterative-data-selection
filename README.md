# data-selection

## Dataset Specifications
Check share documentation at Google Docs

## Finetuning

### Scripts
```bash
# Llama-2-7b-hf (with accelerate and deepspeed)
bash scripts/finetune_llama_with_accelerate.sh [SELECTION_METHOD]
```

## Coreset Selection
The hyperparameters and configurations are managed by [Hydra](https://hydra.cc/). The configurations are stored in `selection/config/`.
You should run the code by executing `main.py` in the `selection` directory. You can also specify the hyperparameters by command line arguments.
```bash
cd selection
python main.py data=[sharegpt|wizardlm] encoder=miniLM coreset=random
```
The selected indices are stored under `selection/indices/`.

## Directory Structure
```
.
├── README.md
├── __init__.py
├── analysis  <- Analysis of the results
├── data      <- where the data is stored
├── ds_configs <- DeepSpeed configurations
├── eval       <- Evaluation code
├── finetune   <- Finetuning pipeline and code
├── logs       <- Logs
├── requirement.txt <- Requirements
├── scripts    <- Scripts for finetuning
├── selection  <- Coreset selection code
    ├── __init__.py
    ├── config  <- Hydra configuration files
    ├── encode.py
    ├── encoder <- Encoder code
    ├── indices <- Indices of the selected coreset
    ├── main.py <- Main entry point for selection
    ├── methods <- Coreset selection methods
    ├── p3_plot.ipynb
    ├── plot.py
    ├── plots
    ├── scoring
    ├── sharegpt_plot.ipynb
    └── wizardlm_plot.ipynb
```
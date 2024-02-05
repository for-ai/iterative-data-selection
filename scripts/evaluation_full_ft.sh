export CUDA_VISIBLE_DEVICES=0
HF_TOKEN=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

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
EVAL_BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=64
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf
DATASET_FILE=simonycl/p3_0.5_dataset
TRAIN_FILE=data/processed/sharegpt/sharegpt_data.jsonl
STEP_NAME=""
EPOCH_NAME=""

MODEL_NAME=Llama-2-7b-hf-sharegpt-KCenterGreedy-005-full-ft

cd /mnt/data/lm-evaluation-harness
bash eval_model.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-KCenterGreedy-005-full-ft data_selection_$MODEL_NAME > /mnt/data/EasyLM/eval_results/data_selection_$MODEL_NAME.log

cd /mnt/data/data-selection/
bash scripts/eval/cohere.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-KCenterGreedy-005-full-ft data_selection_$MODEL_NAME > /mnt/data/EasyLM/eval_results/data_selection_$MODEL_NAME-mmlu.log

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KMenasRandomDeita-64-005-lora-epoch_4.log 2>&1 &

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KCenterGreedyDeita-005-lora-epoch_4.log 2>&1 &
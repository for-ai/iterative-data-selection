export CUDA_VISIBLE_DEVICES=0,1
HF_TOKEN=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

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
EVAL_BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=64
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf
DATASET_FILE=simonycl/p3_0.5_dataset
TRAIN_FILE=data/processed/sharegpt/sharegpt_data.jsonl
STEP_NAME=step_3000
MODEL_NAME=Llama-2-7b-hf-sharegpt
# MODEL_NAME=Llama-2-7b-hf-lima

python3 finetune/merge_lora.py \
    --base_model_name_or_path $MODEL_NAME_OR_PATH \
    --lora_model_name_or_path output/data_selection_${MODEL_NAME}_lora/${STEP_NAME} \
    --output_dir output/data_selection_${MODEL_NAME}_lora_merged_${STEP_NAME}/ \
    --tokenizer_name_or_path output/data_selection_${MODEL_NAME}_lora/ \
    --push_to_hub_id simonycl/data_selection_${MODEL_NAME}_lora_merged_${STEP_NAME} \
    --save_tokenizer
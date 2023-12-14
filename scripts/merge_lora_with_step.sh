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
STEP_NAME=""
EPOCH_NAME=""

MODEL_NAME=Llama-2-7b-hf-p3-0.5-test
# MODEL_NAME=Llama-2-7b-hf-lima

for EPOCH_NAME in 3 4 5 6 7 8 
do
    # check if step_name equal to ""
    if [ -z "$STEP_NAME" ] && [ -z "$EPOCH_NAME" ]
    then
        echo "STEP_NAME is empty"
        python3 finetune/merge_lora.py \
        --base_model_name_or_path $MODEL_NAME_OR_PATH \
        --lora_model_name_or_path output/data_selection_${MODEL_NAME}_lora \
        --output_dir output/data_selection_${MODEL_NAME}_lora_merged/ \
        --tokenizer_name_or_path output/data_selection_${MODEL_NAME}_lora/ \
        --push_to_hub_id simonycl/data_selection_${MODEL_NAME}_lora \
        --save_tokenizer
    # check if EPOCH_NAME is not empty
    elif [ -n "$EPOCH_NAME" ]
    then
        echo "EPOCH_NAME is NOT empty"
        echo "EPOCH_NAME is ${EPOCH_NAME}"
        python3 finetune/merge_lora.py \
        --base_model_name_or_path $MODEL_NAME_OR_PATH \
        --lora_model_name_or_path output/data_selection_${MODEL_NAME}_lora/epoch_${EPOCH_NAME} \
        --output_dir output/data_selection_${MODEL_NAME}_lora_merged_epoch_${EPOCH_NAME}/ \
        --tokenizer_name_or_path output/data_selection_${MODEL_NAME}_lora/ \
        --save_tokenizer
    else
        echo "STEP_NAME is NOT empty"
        python3 finetune/merge_lora.py \
        --base_model_name_or_path $MODEL_NAME_OR_PATH \
        --lora_model_name_or_path output/data_selection_${MODEL_NAME}_lora/${STEP_NAME} \
        --output_dir output/data_selection_${MODEL_NAME}_lora_merged_${STEP_NAME}/ \
        --tokenizer_name_or_path output/data_selection_${MODEL_NAME}_lora/ \
        --push_to_hub_id simonycl/data_selection_${MODEL_NAME}_lora_merged_${STEP_NAME} \
        --save_tokenizer
    fi
done
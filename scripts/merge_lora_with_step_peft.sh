export CUDA_VISIBLE_DEVICES=0,1
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

# MODEL_NAME=Llama-2-7b-hf-p3-0.1-kmeansrandom
# MODEL_NAME=Llama-2-7b-hf-lima
MODEL_NAME=Llama-2-7b-hf-p3-full

# for EPOCH_NAME in 0
# do
#     # check if step_name equal to ""
#     if [ -z "$STEP_NAME" ] && [ -z "$EPOCH_NAME" ]
#     then
#         echo "STEP_NAME is empty"
#         python3 finetune/merge_lora.py \
#         --base_model_name_or_path $MODEL_NAME_OR_PATH \
#         --lora_model_name_or_path output/data_selection_${MODEL_NAME}_lora \
#         --output_dir output/data_selection_${MODEL_NAME}_lora_merged/ \
#         --tokenizer_name_or_path output/data_selection_${MODEL_NAME}_lora/ \
#         --save_tokenizer
#     # check if EPOCH_NAME is not empty
#     elif [ -n "$EPOCH_NAME" ]
#     then
#         echo "EPOCH_NAME is NOT empty"
#         echo "EPOCH_NAME is ${EPOCH_NAME}"
#         python3 finetune/merge_lora.py \
#         --base_model_name_or_path $MODEL_NAME_OR_PATH \
#         --lora_model_name_or_path simonycl/data-selection-Llama-2-7b-p3-full-epoch-0 \
#         --output_dir output/data_selection_${MODEL_NAME}_lora_merged_epoch_${EPOCH_NAME}/ \
#         --tokenizer_name_or_path $MODEL_NAME_OR_PATH \
#         --save_tokenizer
#     else
#         echo "STEP_NAME is NOT empty"
#         python3 finetune/merge_lora.py \
#         --base_model_name_or_path $MODEL_NAME_OR_PATH \
#         --lora_model_name_or_path output/data_selection_${MODEL_NAME}_lora/${STEP_NAME} \
#         --output_dir output/data_selection_${MODEL_NAME}_lora_merged_${STEP_NAME}/ \
#         --tokenizer_name_or_path output/data_selection_${MODEL_NAME}_lora/ \
#         --push_to_hub_id simonycl/data_selection_${MODEL_NAME}_lora_merged_${STEP_NAME} \
#         --save_tokenizer
#     fi
# done

# for epoch in 0
# do 
#     CHECKPOINT_PATH=output/data_selection_${MODEL_NAME}_lora_merged_epoch_${epoch}
#     DATASET_FILE=simonycl/p3_0.5_dataset
#     python3 -m finetune.evaluation \
#         --use_flash_attn \
#         --max_seq_length 4096 \
#         --dataset_name $DATASET_FILE \
#         --output_dir output/data_selection_${MODEL_NAME}_lora_merged_epoch_${epoch} \
#         --model_name_or_path $CHECKPOINT_PATH \
#         --eval_batch_size 16
# done

for peft in simonycl/data-selection-Llama-2-7b-p3-uniform-upsample-epoch-24
do
    PEFT_PATH=$peft
    # extract the model name from split string by delimiter '-' and get the third last element
    MODEL_NAME=$(echo $PEFT_PATH | cut -d'-' -f 3- | rev | cut -d'-' -f 3- | rev)
    MODEL_NAME=Llama-2-7b-hf-lora-${MODEL_NAME}
    DATASET_FILE=simonycl/p3_0.5_dataset
    python3 -m finetune.evaluation \
        --use_flash_attn \
        --max_seq_length 4096 \
        --dataset_name $DATASET_FILE \
        --output_dir output/data_selection_${MODEL_NAME} \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --use_lora $PEFT_PATH \
        --eval_batch_size 16
done

# nohup bash scripts/merge_lora_with_step_peft.sh > output/p3-random-0.1-lr-3e-5-epoch-19-eval-results.log 2>&1 &
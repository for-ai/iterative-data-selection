export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
EVAL_BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=64
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf
# MODEL_NAME_OR_PATH=output/data_selection_Llama-2-7b-hf-lima_lora_merged
DATASET_FILE=simonycl/p3_0.5_dataset
TRAIN_FILE=data/processed/lima/lima_data.jsonl

MODEL_NAME=Llama-2-7b-hf-lima

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    finetune/finetune.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --use_flash_attn \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 2048 \
    --preprocessing_num_workers 24 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_train_epochs 15 \
    --do_eval \
    --eval_file data/processed/lima/lima_test_data.jsonl \
    --output_dir output/data_selection_${MODEL_NAME}_lora_test \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1

# python3 finetune/merge_lora.py \
#     --base_model_name_or_path $MODEL_NAME_OR_PATH \
#     --lora_model_name_or_path output/data_selection_${MODEL_NAME}_lora \
#     --output_dir output/data_selection_${MODEL_NAME}_lora_merged/ \
#     --push_to_hub_id simonycl/data_selection_${MODEL_NAME}_lora_merged \
#     --save_tokenizer

# nohup bash scripts/finetune_llama_with_accelerate.sh > logs/finetune_with_accelerate_Llama-2-7b-hf-sharegpt_lora_1.log 2>&1 &

# nohup bash scripts/finetune_llama_with_accelerate_lima.sh > logs/finetune_with_accelerate_Llama-2-7b-hf-lima_lora.log 2>&1 &
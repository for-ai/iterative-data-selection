export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=4
EVAL_BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=128
MODEL_NAME_OR_PATH=facebook/galactica-1.3b
DATASET_FILE=simonycl/p3_0.5_dataset

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    src/finetune.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --use_slow_tokenizer \
    --dataset_name $DATASET_FILE \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --do_eval \
    --eval_dataset_name rte \
    --eval_batch_size $EVAL_BATCH_SIZE_PER_GPU \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_train_epochs 2 \
    --output_dir output/data_selection_${MODEL_NAME_OR_PATH}_lora/ \
    --with_tracking
    # --report_to wandb \
    # --logging_steps 1

python3 src/merge_lora.py \
    --base_model_name_or_path $MODEL_NAME_OR_PATH \
    --lora_model_name_or_path output/data_selection_${MODEL_NAME_OR_PATH}_lora/ \
    --output_dir output/data_selection_${MODEL_NAME_OR_PATH}_lora_merged/ \
    --push_to_hub_id simonycl/data_selection_${MODEL_NAME_OR_PATH}_lora_merged \
    --save_tokenizer
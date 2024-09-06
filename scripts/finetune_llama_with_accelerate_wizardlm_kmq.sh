INDICES=KMeansRandomDeita_0.1_text-embedding-3-small

# replace all "_" by "-" in INDICES to INDICES_NAME
INDICES_NAME=${INDICES//_/-}

export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
EVAL_BATCH_SIZE_PER_GPU=16
TOTAL_BATCH_SIZE=64
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf

DATASET=wizardlm

TRAIN_FILE=data/processed/${DATASET}/${DATASET}_data.jsonl

MODEL_NAME=Llama-2-7b-hf-${DATASET}-lora-${INDICES_NAME}

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    finetune/finetune.py \
    --tracker_model_name $MODEL_NAME \
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
    --weight_decay 0.00 \
    --num_train_epochs 3 \
    --do_eval \
    --eval_file data/processed/ultrachat/test_1000.jsonl \
    --eval_steps 100 \
    --eval_batch_size $EVAL_BATCH_SIZE_PER_GPU \
    --selection_indices selection/indices/${DATASET}_${INDICES}.pkl \
    --output_dir output/data_selection_${DATASET}_${MODEL_NAME} \
    --with_tracking \
    --logging_steps 10 \
    --report_to wandb

mkdir -p eval_results
# Run evaluation on ARC, GSM8K, HellaSwag, TruthfulQA, and TrivialQA
bash lm-evaluation-harness/eval_model.sh output/data_selection_${DATASET}_${MODEL_NAME}/ data_selection_$MODEL_NAME > eval_results/data_selection_$MODEL_NAME.log

# Evaluation script for MMLU, TydiQA and CodeX-HumanEval
bash scripts/eval/mmlu.sh output/data_selection_${DATASET}_${MODEL_NAME}/ data_selection_$MODEL_NAME > eval_results/data_selection_$MODEL_NAME-mmlu.log
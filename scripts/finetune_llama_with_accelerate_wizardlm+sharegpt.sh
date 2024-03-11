for INDICES in Random_0.1_text-embedding-3-small KMeansRandom_0.05_text-embedding-3-small_64
do

    # replace all "_" by "-" in INDICES to INDICES_NAME
    INDICES_NAME=${INDICES//_/-}

    export CUDA_VISIBLE_DEVICES=0,1

    MODEL_SIZE=7B
    NUM_GPUS=2
    BATCH_SIZE_PER_GPU=2
    EVAL_BATCH_SIZE_PER_GPU=16
    TOTAL_BATCH_SIZE=64
    MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf

    DATASET=wizardlm+sharegpt

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
        --use_lora \
        --lora_rank 64 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --num_train_epochs 3 \
        --do_eval \
        --eval_file data/processed/ultrachat/test_1000.jsonl \
        --eval_steps 100 \
        --eval_batch_size $EVAL_BATCH_SIZE_PER_GPU \
        --selection_indices selection/indices/${DATASET}_${INDICES}.pkl \
        --output_dir output/data_selection_${DATASET}_${MODEL_NAME}_lora \
        --with_tracking \
        --logging_steps 10 \
        --report_to wandb

    python3 finetune/merge_lora.py \
        --base_model_name_or_path $MODEL_NAME_OR_PATH \
        --lora_model_name_or_path output/data_selection_${MODEL_NAME}_lora \
        --output_dir output/data_selection_${MODEL_NAME}_lora_merged/ \
        --push_to_hub_id simonycl/data_selection_${MODEL_NAME}_lora_merged \
        --save_tokenizer

    mkdir -p eval_results
    # Run evaluation on ARC, GSM8K, HellaSwag, TruthfulQA, and TrivialQA
    bash lm-evaluation-harness/eval_model.sh output/data_selection_${MODEL_NAME}_lora_merged/ data_selection_$MODEL_NAME > eval_results/data_selection_$MODEL_NAME.log

    # Evaluation script for MMLU, TydiQA and CodeX-HumanEval
    bash scripts/eval/mmlu.sh output/data_selection_${MODEL_NAME}_lora_merged/ data_selection_$MODEL_NAME > eval_results/data_selection_$MODEL_NAME-mmlu.log
done

# nohup bash scripts/finetune_llama_with_accelerate.sh > logs/finetune_with_accelerate_Llama-2-7b-hf-sharegpt_lora_1.log 2>&1 &

# nohup bash scripts/finetune_llama_with_accelerate.sh KCenterGreedyDeita_0.05_Llama-2-7b-hf > logs/finetune_with_accelerate_Llama-2-7b-hf-cohere-KCenterGreedyDeita_0.05_lora.log 2>&1 &

# nohup bash scripts/finetune_llama_with_accelerate.sh KMeansDynamic_0.05_Llama-2-7b-hf > logs/finetune_with_accelerate_Llama-2-7b-hf-cohere-lora-KMeansDynamic_0.05.log 2>&1 &
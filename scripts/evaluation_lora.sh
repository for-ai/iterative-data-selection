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

MODEL_NAME=Llama-2-7b-hf-sharegpt-KCenterMedian-0.05-lora-epoch_4

# for peft in simonycl/llama-2-7b-hf-cohere-KCenterGreedyDeita-0.05-Llama-2-7b-hf-2e-5-norm simonycl/llama-2-7b-hf-cohere-KMenasRandomDeita-0.05-Llama-2-7b-hf-2e-5-1024-norm
# for peft in simonycl/llama-2-7b-hf-cohere-KMenasRandomDeita-0.05-Llama-2-7b-hf-2e-5-64-norm simonycl/llama-2-7b-hf-cohere-KMeansDynamic-0.05-Llama-2-7b-hf-2e-5-norm
for peft in simonycl/llama-2-7b-hf-cohere-KMenasRandom-0.05-Llama-2-7b-hf-2e-5-1024-norm
do
    cd /mnt/ceph_rbd/data-selection/
    PEFT_PATH=$peft
    MODEL_NAME=$(echo $PEFT_PATH | tr '/' '-' | cut -d '-' -f 2-) # output: llama-2-7b-hf-cohere-Random-0.05 or llama-2-7b-hf-cohere-KCenterGreedyDeita-0.05-Llama-2-7b-hf
    # extract the model name from split string by delimiter '-' and get the third last element
    python3 -m finetune.merge_lora \
        --lora_model_name_or_path $PEFT_PATH \
        --base_model_name_or_path $MODEL_NAME_OR_PATH \
        --tokenizer_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir /mnt/ceph_rbd/data-selection/output/data_selection_${MODEL_NAME} \
        --save_tokenizer \
        --push_to_hub_id simonycl/data_selection_${MODEL_NAME}

    cd /mnt/ceph_rbd/lm-evaluation-harness
    bash eval_model.sh /mnt/ceph_rbd/data-selection/output/data_selection_${MODEL_NAME} data_selection_$MODEL_NAME > /mnt/ceph_rbd/EasyLM/eval_results/data_selection_$MODEL_NAME.log

    cd /mnt/ceph_rbd/data-selection/
    bash scripts/eval/cohere.sh /mnt/ceph_rbd/data-selection/output/data_selection_${MODEL_NAME} data_selection_$MODEL_NAME > /mnt/ceph_rbd/EasyLM/eval_results/data_selection_$MODEL_NAME-mmlu.log
done

# simonycl/llama-2-7b-hf-cohere-KCenterGreedyDeita-0.05-Llama-2-7b-hf

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KMenasRandomDeita-64-005-lora-epoch_4.log 2>&1 &

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KMenasRandomDeita-512-0.05-lora-epoch_4.log 2>&1 &

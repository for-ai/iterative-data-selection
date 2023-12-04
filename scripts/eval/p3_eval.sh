export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_PATH=/mnt/ceph_rbd/data-selection/output/data_selection_Llama-2-7b-hf-p3_lora_merged
MODEL_NAME=p3-ft-llama-7B
DATASET_FILE=simonycl/p3_0.5_dataset
python3 -m finetune.evaluation \
    --use_flash_attn \
    --max_seq_length 4096 \
    --dataset_name $DATASET_FILE \
    --output_dir output/data_selection_${MODEL_NAME}_lora_merged \
    --model_name_or_path $CHECKPOINT_PATH

CHECKPOINT_PATH=meta-llama/Llama-2-7b-hf
MODEL_NAME=llama-7B
python3 -m finetune.evaluation \
    --use_flash_attn \
    --max_seq_length 4096 \
    --dataset_name $DATASET_FILE \
    --output_dir output/data_selection_${MODEL_NAME}_lora_merged \
    --model_name_or_path $CHECKPOINT_PATH
    
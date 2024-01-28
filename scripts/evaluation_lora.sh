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

MODEL_NAME=Llama-2-7b-hf-sharegpt-KMeansCentroidDeita-1024-005-lora-epoch_4

# for peft in simonycl/data-selection-Llama-2-7b-sharegpt-KMeansCentroidDeita-0.05-lora-epoch-4
# do
#     PEFT_PATH=$peft
#     # extract the model name from split string by delimiter '-' and get the third last element
#     python3 -m finetune.merge_lora \
#         --lora_model_name_or_path $PEFT_PATH \
#         --base_model_name_or_path $MODEL_NAME_OR_PATH \
#         --tokenizer_name_or_path $MODEL_NAME_OR_PATH \
#         --output_dir /mnt/ceph_rbd/data-selection/output/data_selection_${MODEL_NAME} \
#         --save_tokenizer \
#         --push_to_hub_id simonycl/data_selection_${MODEL_NAME}
# done
# 
# cd /mnt/ceph_rbd/lm-evaluation-harness
# bash eval_model.sh /mnt/ceph_rbd/data-selection/output/data_selection_${MODEL_NAME} data_selection_$MODEL_NAME > /mnt/ceph_rbd/EasyLM/eval_results/data_selection_$MODEL_NAME.log

cd /mnt/ceph_rbd/data-selection/
bash scripts/eval/cohere.sh /mnt/ceph_rbd/data-selection/output/data_selection_${MODEL_NAME} data_selection_$MODEL_NAME > /mnt/ceph_rbd/EasyLM/eval_results/data_selection_$MODEL_NAME-mmlu-1.log

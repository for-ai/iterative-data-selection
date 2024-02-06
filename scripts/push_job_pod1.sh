METHOD=KMeansCentroidDeita_0.05

# sleep 4 hours
# sleep 2h

export HF_HOME=/mnt/ceph_rbd/.cache/huggingface
export HF_TRANSFORMERS_CACHE=/mnt/ceph_rbd/.cache/transformers
export HF_DATASETS_CACHE=/mnt/ceph_rbd/.cache/datasets

git config --global credential.helper store

huggingface-cli login --token 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

wandb login 99c1cfcf5ab402b2d7df6da383d1645fe6da06b6

cd /mnt/ceph_rbd/data-selection/output/data_selection_cohere_Llama-2-7b-hf-cohere-lora-KCenterGreedyDeita-0.05-Llama-2-7b-hf_lora/epoch_3

huggingface-cli upload simonycl/llama-2-7b-hf-cohere-KCenterGreedyDeita-0.05-Llama-2-7b-hf-2e-5-epoch-3 . .

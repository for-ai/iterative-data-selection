
export HF_HOME=/mnt/ceph_rbd/.cache/huggingface
export HF_TRANSFORMERS_CACHE=/mnt/ceph_rbd/.cache/transformers
export HF_DATASETS_CACHE=/mnt/ceph_rbd/.cache/datasets

git config --global credential.helper store

huggingface-cli login --token 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

wandb login 99c1cfcf5ab402b2d7df6da383d1645fe6da06b6

gh auth login --with-token <<< "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"

gh auth setup-git
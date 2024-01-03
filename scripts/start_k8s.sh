pip3 install protobuf==4.23.0

pip3 install nvitop

pip3 install accelerate deepspeed peft bitsandbytes tokenizers evaluate

conda install gh -c conda-forge

pip3 install git+https://github.com/yizhongw/transformers.git@left_padding

pip install packaging
pip install ninja

pip install flash-attn --no-build-isolation

pip install wandb

export HF_HOME=/mnt/ceph_rbd/.cache/huggingface
export HF_TRANSFORMERS_CACHE=/mnt/ceph_rbd/.cache/transformers
export HF_DATASETS_CACHE=/mnt/ceph_rbd/.cache/datasets

git config --global credential.helper store

huggingface-cli login --token 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

wandb login 99c1cfcf5ab402b2d7df6da383d1645fe6da06b6
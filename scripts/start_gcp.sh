
sudo apt-get update


sudo apt-get install -y vim

# mount disk
sudo mkfs.ext4 /dev/sdb

sudo mkdir /mnt/data

sudo mount /dev/sdb /mnt/data

cd /mnt/data
# install anaconda
conda activate

conda create -p /mnt/ceph_rbd/selection python==3.10

conda activate /mnt/ceph_rbd/selection

pip3 install torch torchvision torchaudio scipy packaging sentencepiece datasets transformers accelerate wandb

pip3 install protobuf==4.23.0

pip3 install nvitop

pip3 install accelerate deepspeed peft bitsandbytes tokenizers evaluate

conda install gh -c conda-forge

pip3 install git+https://github.com/yizhongw/transformers.git@left_padding

pip install flash-attn --no-build-isolation

pip install wandb

## export the cache
conda activate /mnt/data/selection

export HF_HOME=/mnt/data/.cache/huggingface
export HF_TRANSFORMERS_CACHE=/mnt/data/.cache/transformers
export HF_DATASETS_CACHE=/mnt/data/.cache/datasets

git config --global credential.helper store

huggingface-cli login --token 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

wandb login 99c1cfcf5ab402b2d7df6da383d1645fe6da06b6

type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

gh auth login --with-token <<< "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"
gh auth setup-git
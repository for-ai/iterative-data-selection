METHOD=KMeansCentroidDeita_0.05

# sleep 4 hours
sleep 4h

conda activate /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/conda/multi

export HF_HOME=/home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/huggingface
export HF_TRANSFORMERS_CACHE=/home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/transformers
export HF_DATASETS_CACHE=/home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/datasets

git config --global credential.helper store

huggingface-cli login --token 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

wandb login 99c1cfcf5ab402b2d7df6da383d1645fe6da06b6

cd /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/data-selection/output/data_selection_cohere_Llama-2-7b-hf-cohere-lora-KMeansDynamic-0.05-Llama-2-7b-hf_lora/epoch_4

huggingface-cli upload simonycl/llama-2-7b-hf-cohere-KMeansDynamic-0.05-Llama-2-7b-hf-2e-5 . .

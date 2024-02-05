# sleep 4 hours
sleep 6h

METHOD=KMenasMedianDeita_0.05

conda activate /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/conda/multi

export HF_HOME=/home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/huggingface
export HF_TRANSFORMERS_CACHE=/home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/transformers
export HF_DATASETS_CACHE=/home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/datasets

git config --global credential.helper store

huggingface-cli login --token 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

wandb login 99c1cfcf5ab402b2d7df6da383d1645fe6da06b6

cd /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/data-selection

conda activate /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/conda/multi

curl -X POST -H 'Content-type: application/json' --data '{"text":"Experiment KMenasMedianDeita_0.05 Starting in co-huan1"}' https://hooks.slack.com/services/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

nohup bash scripts/finetune_llama_with_accelerate.sh ${METHOD} > logs/finetune_with_accelerate_Llama-2-7b-hf-sharegpt-${METHOD}_lora.log 2>&1 &
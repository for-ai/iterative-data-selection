export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
    --main_process_port=2950 selection/iter.py \
    --model simonycl/data_selection_wizardlm_sharegpt_iter_0 \
    --output_dir selection/iter_data/wizardlm_sharegpt_iter_0.jsonl \
    --dataset data/processed/wizardlm+sharegpt/wizardlm+sharegpt_data.jsonl \
    --indices selection/indices/wizardlm+sharegpt_KMeansIter_0.05_text-embedding-3-small_round_3_iter_0.pkl \
    --portion 0.5

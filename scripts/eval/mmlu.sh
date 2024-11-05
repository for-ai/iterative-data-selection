export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_PATH=$1
MODEL_NAME=$2

# # MMLU 5 shot
python3 -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/${MODEL_NAME}-5shot \
    --model_name_or_path $CHECKPOINT_PATH \
    --tokenizer_name_or_path $CHECKPOINT_PATH \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# CodeEval
python3 -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/{$MODEL_NAME}_temp_0_8 \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --use_vllm
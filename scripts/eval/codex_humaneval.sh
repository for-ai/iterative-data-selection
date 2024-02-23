# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0,1
export OPENAI_API_KEY=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

# for model in /mnt/ceph_rbd/data-selection/output/data_selection_llama-2-7b-hf-cohere-KMenasRandomDeita-0.05-Llama-2-7b-hf-2e-5-1024-norm
for model in /mnt/ceph_rbd/data-selection/output/data_selection_llama-2-7b-hf-cohere-KCenterGreedyDeita-0.05-Llama-2-7b-hf-2e-5-norm
do
    model_name=$(echo $model | tr '/' '-' | cut -d '-' -f 2-) # output: llama-2-7b-hf-cohere-Random-0.05
    # Evaluating llama 7B model using temperature 0.1 to get the pass@1 score
    # python3 -m eval.codex_humaneval.run_eval \
    #     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    #     --eval_pass_at_ks 1 5 10 20 \
    #     --unbiased_sampling_size_n 20 \
    #     --temperature 0.1 \
    #     --save_dir results/codex_humaneval/{$model_name}_temp_0_1 \
    #     --model $model \
    #     --tokenizer $model \
    #     --use_vllm


    # Evaluating llama 7B model using temperature 0.8 to get the pass@10 score
    python3 -m eval.codex_humaneval.run_eval \
        --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
        --eval_pass_at_ks 10 \
        --unbiased_sampling_size_n 20 \
        --temperature 0.8 \
        --save_dir results/codex_humaneval/{$model_name}_temp_0_8 \
        --model $model \
        --tokenizer $model \
        --use_vllm
done

# Evaluating tulu 7B model using temperature 0.1 to get the pass@1 score
# We don't use chat format for codex_humaneval, since it's not a chat dataset
# But you can use it by adding --use_chat_format and --chat_formatting_function create_prompt_with_tulu_chat_format
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/tulu_7B_temp_0_1 \
#     --model ../checkpoints/tulu_7B/ \
#     --tokenizer ../checkpoints/tulu_7B/ \
#     --use_vllm


# # Evaluating tulu 7B model using temperature 0.8 to get the pass@10 score
# # We don't use chat format for codex_humaneval, since it's not a chat dataset
# # But you can use it by adding --use_chat_format and --chat_formatting_function create_prompt_with_tulu_chat_format
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
#     --eval_pass_at_ks 10 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.8 \
#     --save_dir results/codex_humaneval/tulu_7B_temp_0_8 \
#     --model ../checkpoints/tulu_7B/ \
#     --tokenizer ../checkpoints/tulu_7B/ \
#     --use_vllm


# # Evaluating chatgpt using temperature 0.1 to get the pass@1 score
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --save_dir results/codex_humaneval/chatgpt_temp_0.1/ \
#     --eval_batch_size 10


# # Evaluating chatgpt using temperature 0.8 to get the pass@10 score
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.8 \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --save_dir results/codex_humaneval/chatgpt_temp_0.8/ \
#     --eval_batch_size 10


# # Evaluating gpt4 using temperature 0.1 to get the pass@1 score
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --openai_engine "gpt-4-0314" \
#     --save_dir results/codex_humaneval/gpt4_temp_0.1 \
#     --eval_batch_size 1


# # Evaluating gpt4 using temperature 0.8 to get the pass@10 score
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.8 \
#     --openai_engine "gpt-4-0314" \
#     --save_dir results/codex_humaneval/gpt4_temp_0.8 \
#     --eval_batch_size 1

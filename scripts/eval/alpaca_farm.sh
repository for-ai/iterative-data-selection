# Please make sure OPENAI_API_KEY is set in your environment variables
export CUDA_VISIBLE_DEVICES=0,1
export OPENAI_API_KEY=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

# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --save_dir results/alpaca_farm/llama-7b/ \
#     --eval_batch_size 32 \
#     --use_vllm \
#     --max_new_tokens 1024
#     # --use_chat_format \
#     # --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --save_dir results/alpaca_farm/llama-7b-chat/ \
#     --eval_batch_size 32 \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format

# use vllm for generation
# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
#     --save_dir results/alpaca_farm/tulu_v1_7B/ \
#     --eval_batch_size 32 \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

python -m eval.alpaca_farm.run_eval \
    --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
    --save_dir results/alpaca_farm/lima_ft_7B/ \
    --eval_batch_size 32 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# # use normal huggingface generation function
# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
#     --save_dir results/alpaca_farm_hf/tulu_v1_7B/ \
#     --eval_batch_size 16 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --load_in_8bit
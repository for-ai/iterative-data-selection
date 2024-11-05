# Please make sure OPENAI_API_KEY is set in your environment variables
export CUDA_VISIBLE_DEVICES=0,1

python -m eval.alpaca_farm.run_eval \
    --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
    --save_dir results/alpaca_farm/lima_ft_7B/ \
    --eval_batch_size 32 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
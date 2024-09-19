# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0,1


# Evaluating llama 7B model using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-cot-8shot \
#     --model meta-llama/Llama-2-7b-hf \
#     --tokenizer meta-llama/Llama-2-7b-hf \
#     --n_shot 8 \
#     --use_vllm


# # Evaluating llama 7B model using direct answering (no chain-of-thought)
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-no-cot-8shot \
#     --model meta-llama/Llama-2-7b-hf \
#     --tokenizer meta-llama/Llama-2-7b-hf \
#     --n_shot 8 \
#     --no_cot \
#     --use_vllm

# Evaluating llama 7B model using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-chat-cot-0shot \
#     --model meta-llama/Llama-2-7b-chat-hf \
#     --tokenizer meta-llama/Llama-2-7b-chat-hf \
#     --n_shot 0 \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \

# # Evaluating llama 7B model using direct answering (no chain-of-thought)
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-chat-no-cot-0shot \
#     --model meta-llama/Llama-2-7b-chat-hf \
#     --tokenizer meta-llama/Llama-2-7b-chat-hf \
#     --n_shot 0 \
#     --no_cot \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \

# # Evaluating sharegpt tuned LLaMA 7B model using chain-of-thought and chat format


# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/lima-ft-llama-7B-cot-0shot \
#     --model /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged\
#     --tokenizer /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
#     --n_shot 0 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm

# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/sharegpt-ft-llama-7B-cot-0shot-2000 \
#     --model /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000\
#     --tokenizer /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
#     --n_shot 0 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm

# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/sharegpt-ft-llama-7B-cot-0shot-2400 \
#     --model /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2400 \
#     --tokenizer /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2400 \
#     --n_shot 0 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm

# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/sharegpt-ft-llama-7B-cot-0shot-3600 \
#     --model /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_3600 \
#     --tokenizer /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_3600 \
#     --n_shot 0 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm


# # 5 shot version

python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/lima-ft-llama-7B-cot-5shot \
    --model /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged\
    --tokenizer /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
    --n_shot 5 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/sharegpt-ft-llama-7B-cot-5shot-2000 \
    --model /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000\
    --tokenizer /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
    --n_shot 5 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/sharegpt-ft-llama-7B-cot-5shot-2400 \
    --model /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2400 \
    --tokenizer /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2400 \
    --n_shot 5 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/sharegpt-ft-llama-7B-cot-5shot-3600 \
    --model /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_3600 \
    --tokenizer /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_3600 \
    --n_shot 5 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm
    

# # Evaluating llama2 chat model using chain-of-thought and chat format
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama2-chat-7B-cot-8shot \
#     --model ../hf_llama2_models/7B-chat \
#     --tokenizer ../hf_llama2_models/7B-chat \
#     --n_shot 8 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
#     --use_vllm


# # Evaluating chatgpt using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/chatgpt-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 8 


# # Evaluating chatgpt using direct answering (no chain-of-thought)
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/chatgpt-no-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 8 \
#     --no_cot


# # Evaluating gpt4 using chain-of-thought
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/gpt4-cot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --n_shot 8 


# # Evaluating gpt4 using direct answering (no chain-of-thought)
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/gpt4-no-cot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --n_shot 8 \
#     --no_cot

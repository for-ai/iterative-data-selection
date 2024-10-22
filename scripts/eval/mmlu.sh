export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_PATH=$1
MODEL_NAME=$2
# CHECKPOINT_PATH=/mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged
# MODEL_NAME=sharegpt-ft-llama-7B
export OPENAI_API_KEY=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

# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/${MODEL_NAME}-0shot-wo-chat \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --eval_batch_size 16 \
#     --load_in_8bit

# # # MMLU 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/${MODEL_NAME}-5shot-wo-chat \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --eval_batch_size 4 \
#     --load_in_8bit

#     
# python -m eval.mmlu.run_eval \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --eval_batch_size 16 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

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

# Evaluating llama 7B model using 0 shot directly
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B-0shot \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
#     --eval_batch_size 16 \
#     --load_in_8bit


# Evaluating llama 7B model using 5 shot directly
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B-5shot \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
#     --eval_batch_size 4 \
#     --load_in_8bit


# # # Evaluating Tulu 7B model using 0 shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/tulu-7B-0shot \
#     --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
#     --tokenizer_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
#     --eval_batch_size 16 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating ShareGPT 7B model using 5 shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/tulu-7B-5shot \
#     --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
#     --tokenizer_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/sharegpt-ft-7B-5shot-step-2000 \
#     --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
#     --tokenizer_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000 \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/sharegpt-ft-7B-5shot-step-3600 \
#     --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_3600 \
#     --tokenizer_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_3600 \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/lima-ft-llama-7B-cot-5shot \
#     --model /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
#     --tokenizer /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
#     --n_shot 5 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     --use_vllm

# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/lima-ft-7B-0shot \
#     --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
#     --tokenizer_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# # 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/lima-ft-7B-5shot \
#     --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
#     --tokenizer_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# python -m eval.truthfulqa.run_eval \
#     --data_dir data/eval/truthfulqa \
#     --save_dir results/trutufulqa/lima-ft-7B-0shot \
#     --model_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
#     --tokenizer_name_or_path /mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged \
#     --metrics judge info mc \
#     --preset qa \
#     --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
#     --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# # Evaluating llama2 chat model using 0-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating llama2 chat model using 5-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating chatgpt using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-0shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating chatgpt using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-5shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating gpt4 using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-0shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20


# # Evaluating gpt4 using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-5shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20

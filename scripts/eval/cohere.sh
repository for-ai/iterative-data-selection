export CUDA_VISIBLE_DEVICES=0,1
# CHECKPOINT_PATH=/mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-lima_lora_merged
# CHECKPOINT_PATH=meta-llama/Llama-2-7b-hf
CHECKPOINT_PATH=/mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_2000
CHECKPOINT_PATH=/mnt/data/data-selection/output/data_selection_Llama-2-7b-hf-sharegpt_lora_merged_step_3000

# MODEL_NAME=lima-ft-llama-7B
# MODEL_NAME=llama-7B
MODEL_NAME=sharegpt-3000-ft-llama-7B
# MODEL_NAME=sharegpt-ft-llama-7B
export OPENAI_API_KEY=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

## GSM8K 8 shot
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/${MODEL_NAME}-cot-8shot \
    --model $CHECKPOINT_PATH \
    --tokenizer $CHECKPOINT_PATH \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm

# ## MMLU 0 shot
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/${MODEL_NAME}-0shot \
    --model_name_or_path $CHECKPOINT_PATH \
    --tokenizer_name_or_path $CHECKPOINT_PATH \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# # MMLU 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/${MODEL_NAME}-5shot \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# # TruthfulQA
# python -m eval.truthfulqa.run_eval \
#     --data_dir data/eval/truthfulqa \
#     --save_dir results/trutufulqa/${MODEL_NAME} \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --metrics judge info mc \
#     --preset qa \
#     --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
#     --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# # ARC Challenge
# python -m eval.arc.run_eval \
#     --ntrain 25 \
#     --save_dir results/arc/${MODEL_NAME} \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --eval_batch_size 8 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # HellaSwag
# python -m eval.hellaswag.run_eval \
#     --ntrain 10 \
#     --save_dir results/hellaswag/${MODEL_NAME} \
#     --model_name_or_path $CHECKPOINT_PATH \
#     --tokenizer_name_or_path $CHECKPOINT_PATH \
#     --eval_batch_size 8 \
#     --n_instances 200 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
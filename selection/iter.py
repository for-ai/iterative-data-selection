from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from collections import Counter


kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

import json

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import pickle as pkl
import torch
import os
from typing import List, Tuple, Dict, Any
from tqdm import trange

# model_path = "meta-llama/Llama-2-7b-hf"
# adapter_path = "simonycl/llama-2-7b-hf-cohere-KMeansIter-0.1-Llama-2-7b-hf-round-4-iter-0"
# dataset_path = '/mnt/data/data-selection/data/processed/cohere/cohere_data.jsonl'
# indices_path = '/mnt/data/data-selection/selection/indices/cohere_KMeansIter_0.1_Llama-2-7b-hf_round_4_iter_0.pkl'

is_vllm = False
BATCH_SIZE = 2

# filter if more than one round of conversation

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--adapter', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='selection/iter_data/cohere_KMeansIter_0.1_Llama-2-7b-hf_round_3_iter_0.jsonl')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='data/processed/cohere/cohere_data.jsonl')
    parser.add_argument('--indices', type=str, default='selection/indices/cohere_KMeansIter_0.1_Llama-2-7b-hf_round_3_iter_0.pkl')
    parser.add_argument('--portion', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--use_rm', action='store_true')
    parser.add_argument('--reward_model', type=str, choices=["ppl", "chatgpt", "raft"])
    return parser.parse_args()

def encode_messages_input_output(messages):
    input = ""
    output = ""
    seen_user = False
    for message in messages:
        if message['role'] == 'system':
            input = '<|system|>\n' + message['content'].strip() + '\n'
        elif message['role'] == 'user':
            if seen_user:
                break
            input += '<|user|>\n' + message['content'].strip() + '\n'
            seen_user = True
        elif message['role'] == 'assistant':
            output += message['content'].strip() + '\n'
    input += '<|assistant|>\n'
    return {
        'input': input,
        'output': output
    }

def generate_with_peft(tokenizer, model, input_outputs, num_return_sequences=1):
    # input_output is a dict with keys 'input' and 'output'
    inputs = [input_output['input'] for input_output in input_outputs]
    # input = input_output['input']
    tokenized_input = tokenizer(
        inputs, 
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=2048,
        ).to(model.device) # (batch_size, max_length)
    
    with torch.no_grad():
        if num_return_sequences == 1:
            output = model.generate(**tokenized_input, max_new_tokens=512, num_return_sequences=num_return_sequences, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        else:
            output = model.generate(**tokenized_input, max_new_tokens=512, num_return_sequences=num_return_sequences, use_cache=True, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    
    decoded_output = tokenizer.batch_decode(output[:, tokenized_input['input_ids'].shape[1]:], skip_special_tokens=True)
    
    if num_return_sequences > 1:
        decoded_output = [[decoded_output[i*num_return_sequences + j] for j in range(num_return_sequences)] for i in range(len(input_outputs))]
    return [
        {
            'input': input_outputs[i]['input'],
            'human': input_outputs[i]['output'],
            'generated': decoded_output[i]
        }
        for i in range(len(input_outputs))
    ]

def encode_for_reward_model(generated):
    '''
    {
    'input': input,
    'human' gold response,
    'generated': generated response
    }
    '''
    input, gold, generated = generated['input'], generated['human'], generated['generated']
    input = input.strip().replace('<|system|>\n', '').replace('<|user|>\n', '').replace('<|assistant|>', '')
    if type(generated) != list:
        generated = [generated]
    generated = [gold] + generated
    reward_messages = []
    for message in generated:
        reward_messages.append(f'###Human: {input} ###Assistant: {message.strip()}')
    return reward_messages

def apply_rm(reward_messages, rm_pipe, pipe_kwargs):
    pipe_outputs = rm_pipe(reward_messages, **pipe_kwargs)
    rewards = [output[0]['score'] for output in pipe_outputs]
    return rewards
# subset_df['input_output'] = subset_df['messages'].apply(encode_messages_input_output)
# subset_df.head()

# merge the encode_for_reward_model and apply_rm functions
def generate_rewards(generated, rm_pipe, pipe_kwargs):
    reward_messages = encode_for_reward_model(generated)
    rewards = apply_rm(reward_messages, rm_pipe, pipe_kwargs)
    return rewards

def rm_pipeline(output_dir):
    rm_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")

    rm_pipe = pipeline(
        "sentiment-analysis",
        model="weqweasdas/hh_rlhf_rm_open_llama_3b",
        device_map={"": accelerator.process_index}, # {"cuda:0": 0, "cuda:1": 1, "cuda:2": 2, "cuda:3": 3
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1
    }

    generated_contents = load_dataset('json', data_files=output_dir, split='train')
    generated_contents = generated_contents.to_list()

    with accelerator.split_between_processes(generated_contents) as generated:
        rewards = []
        for i in trange(len(generated)):
            reward = generate_rewards(generated[i], rm_pipe, pipe_kwargs)
            generated[i]['reward'] = reward
        rewards.extend(generated)

    accelerator.wait_for_everyone()
    rewards_gathered = gather_object(rewards)

    reward_output_dir = output_dir.replace('.jsonl', '_rewards.jsonl')
    with open(reward_output_dir, "w") as f:
        for i, content in enumerate(rewards_gathered):
            f.write(json.dumps(content) + "\n")

    return rewards_gathered

def chatgpt_pipeline(output_dir):
    pass

def calculate_ppl(tokenizer, model, batches):
    tokenized_inputs = tokenizer(
        [batch['input'] + batch['output'] for batch in batches],
        return_tensors="pt", 
        padding="longest", 
        truncation=True, 
        max_length=2048,
        ).to(model.device) # (batch_size, max_length)

    with torch.no_grad():
        outputs = model(**tokenized_inputs, labels=tokenized_inputs['input_ids'])
        loss = outputs.loss
        ppl = torch.exp(loss)
    return ppl.cpu().tolist()

def select_new_iter(rewards_gathered, dataset, indices_path, calculate_method="exp_reward_diff"):
    '''
    {
        'indices': # the indices needed to be selected
    'selected_indices': # the indices selected
    'clusters' # each data belongs to which cluster
    'K' # number of clusters
    'round',
    'iter'
    'portion'
    }
    '''
    with open(indices_path, 'rb') as f:
        indices_detail = pkl.load(f)
    indices = indices_detail['indices']
    clusters = indices_detail['clusters']
    K, round, iter, portion = indices_detail['K'], indices_detail['round'], indices_detail['iter'], indices_detail['portion']
    selected_indices = indices_detail['selected_indices']

    rewards_df = pd.DataFrame(rewards_gathered)
    if calculate_method == "ppl":
        rewards_df['reward_diff'] = rewards_df['reward']
    else:
        rewards_df[['human_reward', 'generated_reward']] = pd.DataFrame(rewards_df['reward'].tolist(), index=rewards_df.index)
        rewards_df = rewards_df.drop(columns=['reward'])
        rewards_df['reward_diff'] = rewards_df['human_reward'] - rewards_df['generated_reward']

    dataset = dataset.add_column('cluster', clusters)
    dataset = dataset.add_column('index', range(len(dataset)))
    subset = dataset.select(indices)
    subset_df = subset.to_pandas()
    subset_df['cluster'] = clusters[indices]

    merged_df = subset_df.merge(rewards_df, left_index=True, right_index=True)
    merged_df = merged_df.groupby('cluster')['reward_diff'].mean().reset_index()
    if calculate_method == "ppl":
        merged_df['exp_reward_diff'] = merged_df['reward_diff']
    else:
        merged_df['exp_reward_diff'] = np.exp(merged_df['reward_diff'])
        merged_df['exp_reward_diff'] = merged_df['exp_reward_diff'] / merged_df['exp_reward_diff'].sum()
    
    size = (len(dataset) * portion) / K / round
    exp_reward_diff = merged_df['exp_reward_diff']
    
    select_new_iter = np.random.choice(K, size=int(size), p=exp_reward_diff, replace=True)
    selected_clusters_size = Counter(select_new_iter)

    remaining_dataset = dataset.select(set(range(len(dataset))) - set(selected_indices))
    remaining_dataset_df = remaining_dataset.to_pandas()

    new_indices = []
    for i in range(K):
        indices = remaining_dataset_df[remaining_dataset_df['cluster'] == i]['index']
        size = min(selected_clusters_size[i], len(indices))
        indices = np.random.choice(indices, size=size, replace=False)
        new_indices.extend(indices)
    new_indices = np.array(new_indices)
    new_indices = np.concatenate([selected_indices, new_indices])

    indices_detail['indices'] = new_indices
    indices_detail['selected_indices'] = new_indices
    indices_detail['iter'] += 1
    indices_detail['portion'] = portion
    new_indices_path = indices_path.replace(f'iter_{iter}', f'iter_{iter+1}')
    with open(new_indices_path, 'wb') as f:
        pkl.dump(indices_detail, f)
    return new_indices_path

def main():
    args = parse_arguments()
    model_path = args.model
    adapter_path = args.adapter
    dataset_path = args.dataset
    indices_path = args.indices
    output_dir = args.output_dir

    if args.use_rm:
        rewards_gathered = rm_pipeline(output_dir)
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
        device_map={"": accelerator.process_index}, # {"cuda:0": 0, "cuda:1": 1, "cuda:2": 2, "cuda:3": 3
        cache_dir="/mnt/data/.cache/huggingface/hub",
    )
    num_added_tokens = tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    })
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    # peft_model = peft_model.to(model.device)

    dataset = load_dataset('json', data_files=dataset_path, split='train')
    with open(indices_path, 'rb') as f:
        indices = pkl.load(f)['indices']
    subset = dataset.select(indices)
    if args.portion < 1.0:
        subset = subset.select(range(int(len(subset) * args.portion)))
    elif args.portion > 1.0:
        subset = subset.select(range(int(args.portion)))

    messages = subset['messages']
    input_outputs = [encode_messages_input_output(message) for message in messages] # list of dicts

    accelerator.wait_for_everyone()

    if args.reward_model == "ppl":
        with accelerator.split_between_processes(input_outputs) as subset:
            perplexities = []

            for i in trange(0, len(subset), args.batch_size):
                batches = subset[i:i+args.batch_size]
                perplexity = calculate_ppl(tokenizer, model, batches)
                perplexities.extend(perplexity)
        
        accelerator.wait_for_everyone()

        perplexities_gathered = gather_object(perplexities)
        print(f"Saving perplexities to {output_dir}")
        if accelerator.is_main_process:
            with open(output_dir, "w") as f:
                for i, perplexity in enumerate(perplexities_gathered):
                    f.write(json.dumps({
                        'input': input_outputs[i]['input'],
                        'human': input_outputs[i]['output'],
                        'ppl': perplexity
                    }) + "\n")
        perplexities_gathered = [{"reward": ppl} for ppl in perplexities_gathered]
    else:
        with accelerator.split_between_processes(input_outputs) as subset:
            generated_content = []

            for i in trange(0, len(subset), args.batch_size):
                batches = subset[i:i+args.batch_size]
                generated = generate_with_peft(tokenizer, model, batches, num_return_sequences=args.num_return_sequences)
                generated_content.extend(generated)

        accelerator.wait_for_everyone()
        
        generated_content_gathered = gather_object(generated_content)

        print(f"Saving generated content to {output_dir}")
        if accelerator.is_main_process:
            with open(output_dir, "w") as f:
                for content in generated_content_gathered:
                    f.write(json.dumps(content) + "\n")

    accelerator.wait_for_everyone()

    print(f"Finished generating rewards for {output_dir}")

    print(f"Generating rewards for {output_dir}")
    if args.reward_model == "raft":
        rewards_gathered = rm_pipeline(output_dir)
    elif args.reward_model == "chatgpt":
        rewards_gathered = chatgpt_pipeline(output_dir)
    elif args.reward_model == "ppl":
        rewards_gathered = perplexities_gathered

    print(f"Selecting new indices for {output_dir}")
    new_indices_path = select_new_iter(rewards_gathered, dataset, indices_path)

if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=1 python selection/encode.py
from encoder import AutoEncoder, get_default_conv_template, concat_tulu_messages
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset
import numpy as np

data_column = 'messages'
model = 'meta-llama/Llama-2-7b-hf'
concat_method = "tulu" # tulu or tulu_v1

def concat_messages(messages, concat_method):
    if concat_method == "tulu":
        return concat_tulu_messages(messages)
    elif concat_method == "tulu_v1":
        template = get_default_conv_template(concat_method)
        for message in messages:
            role, content = message["role"], message["content"]
            template.append_message(role, content)
        return template.get_prompt()
    else:
        raise ValueError(f"Invalid concat method: {concat_method}")
    
def extract_embeddings(dataset, model, concat_method):
    config = {
        "model_name": model,
        'use_flash_attn': True,
        'is_8bit': False,
        'batch_size': 4,
    }
    model = AutoEncoder(config)
    messages = dataset[data_column]
    sentences = [concat_messages(message, concat_method) for message in tqdm(messages, desc="Concatenating messages")]
    embeddings = model.encode(sentences, batch_size=8, device='cuda:1', show_progress_bar=True)
    return embeddings
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/mnt/data/data-selection/data/processed/wizardlm/wizardlm_data.jsonl')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--concat_method', type=str, default='tulu_v1')
    parser.add_argument('--output', type=str, default='/mnt/data/data-selection/data/processed/wizardlm/embeddings.npy')
    args = parser.parse_args()
    
    dataset = load_dataset('json', data_files=args.dataset, split='train')
    # select top 100
    dataset = dataset.select(range(1000))
    embeddings = extract_embeddings(dataset, args.model, args.concat_method)
    np.save(args.output, embeddings)

if __name__ == "__main__":
    main()
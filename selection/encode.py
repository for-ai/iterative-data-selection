# CUDA_VISIBLE_DEVICES=1 python selection/encode.py
from encoder import AutoEncoder, get_default_conv_template, concat_tulu_messages, concat_tulu_messages_only_user
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

def get_dataset(data_config: DictConfig):
    dataset_path = data_config.name
    if dataset_path.endswith('json') or dataset_path.endswith('jsonl'):
        dataset = load_dataset('json', data_files=dataset_path)
    elif dataset_path.endswith('csv'):
        dataset = load_dataset('csv', data_files=dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    if 'split' in data_config:
        dataset = dataset[data_config.split]

    if 'seed' in data_config:
        dataset = dataset.shuffle(seed=data_config.seed)
    if 'subsample' in data_config:
        dataset = dataset.select(range(int(data_config.subsample * len(dataset))))
        
    dataset_name = dataset_path.split('/')[-1].split('_')[0]
    return dataset, dataset_name

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    '''
    '''
    dataset, dataset_name = get_dataset(cfg.data)
    encoder_config = cfg.encoder
    encoder = AutoEncoder(encoder_config)
    embeddings = encoder.get_embeddings(dataset, cfg.data)

if __name__ == "__main__":
    main()
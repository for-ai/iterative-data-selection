"""
Select data subset from a HF dataset using a coreset selection method.
"""
import os
os.environ['LD_LIBRARY_PATH'] = '/mnt/data/selection/lib:' + os.environ['LD_LIBRARY_PATH']

from datasets import load_dataset
import methods
import pickle
import argparse
import os
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

    dataset_name = dataset_path.split('/')[-1].split('_')[0]
    return dataset, dataset_name

# Initialize selector according to yml

def get_subset_indices(dataset, data_config, method_config, encoder_config):
    method_name = method_config.name
    if method_name == "KMenasRandomDeita":
        K = method_config.K
        selector = methods.__dict__[method_name](dataset, data_config=data_config, method_config=method_config, encoder_config=encoder_config, K=K)
    else:
        selector = methods.__dict__[method_name](dataset, data_config=data_config, method_config=method_config, encoder_config=encoder_config)
    subset_indices = selector.select()

    return subset_indices


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    '''
    cfg: DictConfig
    cfg.data: DictConfig
    cfg.coreset: DictConfig
    '''

    dataset, dataset_name = get_dataset(cfg.data)
    method_name = cfg.coreset.name
    fraction = cfg.coreset.fraction
    encoder_config = cfg.encoder if 'encoder' in cfg else None

    subset_indices = get_subset_indices(dataset, cfg.data, cfg.coreset, encoder_config)
    print(f"Selecting {method_name} subset of size {fraction}...")

    output_name = f'selection/indices/{dataset_name}_{method_name}_{str(fraction)}.pkl'
    if encoder_config is not None:
        encoder_name = encoder_config.model_name
        if '/' in encoder_name:
            encoder_name = encoder_name.split('/')[-1]
        output_name = output_name.replace('.pkl', f'_{encoder_name}.pkl')
    if method_name == "KMenasRandomDeita":
        K = cfg.coreset.K
        output_name = output_name.replace('.pkl', f'_{str(K)}.pkl')

    # TODO remove this remember
    output_name = output_name.replace('.pkl', f'_norm.pkl')

    print(f"Saving indices to {output_name}...")
    with open(output_name, 'wb') as f:
        pickle.dump(subset_indices, f)

if __name__ == "__main__":
    main()
# nohup python main.py > logs/main.log 2>&1 &
# export LD_LIBRARY_PATH=/mnt/data/selection/lib:$LD_LIBRARY_PATH

# CUDA_VISIBLE_DEVICES=0 nohup python selection/main.py --multirun data=wizardlm,sharegpt coreset=KMenasRandomDeita encoder=llama > logs/llama.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python selection/main.py --multirun data=wizardlm,sharegpt coreset=KMenasRandomDeita encoder=multilingual-e5,miniLM > logs/multilingual-e5.log 2>&1 &
# python selection/main.py data=cohere coreset=KMeansDynamic encoder=llama
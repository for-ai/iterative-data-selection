"""
Select data subset from a HF dataset using a coreset selection method.
"""

from datasets import load_dataset
import methods
import yaml
import pickle
import argparse
import os

os.environ['LD_LIBRARY_PATH'] = '/mnt/data/selection/lib:$LD_LIBRARY_PATH'

def read_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='p3_config', help='Path to config file')
args = parser.parse_args()


config = read_config(f'./config/{args.config}.yaml')
# Load dataset
dataset_path = config['dataset']['name']
if dataset_path.endswith('json') or dataset_path.endswith('jsonl'):
    dataset = load_dataset('json', data_files=dataset_path)
elif dataset_path.endswith('csv'):
    dataset = load_dataset('csv', data_files=dataset_path)
else:
    dataset = load_dataset(dataset_path)

dataset_name = dataset_path.split('/')[-1].split('_')[0]

if 'split' in config['dataset']:
    dataset = dataset[config['dataset']['split']]
    # dataset = dataset.shuffle()

# Initialize selector according to yml
method_name = config['coreset_method']['name']
fraction = config['coreset_method']['args']['fraction']
if method_name == "KMenasRandomDeita":
    K = config['coreset_method']['args']['K']
    selector = methods.__dict__[method_name](dataset, dataset_config=config['dataset']['args'], method_config=config['coreset_method']['args'], K=K)
else:
    selector = methods.__dict__[method_name](dataset, dataset_config=config['dataset']['args'], method_config=config['coreset_method']['args'])

print(f"Selecting {method_name} subset of size {fraction}...")
# Select subset
subset_indices = selector.select()

if method_name == "KMenasRandomDeita":
    with open(f'indices/{dataset_name}_{method_name}_{str(fraction)}_{str(K)}.pkl', 'wb') as f:
        pickle.dump(subset_indices, f)
else:
    with open(f'indices/{dataset_name}_{method_name}_{str(fraction)}.pkl', 'wb') as f:
        pickle.dump(subset_indices, f)

# nohup python main.py > logs/main.log 2>&1 &
# export LD_LIBRARY_PATH=/mnt/data/selection/lib:$LD_LIBRARY_PATH
"""
Select data subset from a HF dataset using a coreset selection method.
"""

from datasets import load_dataset
import methods
import yaml
import pickle

   
def read_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
    
config = read_config('./config/p3_config.yaml')
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
selector = methods.__dict__[method_name](dataset, dataset_config=config['dataset']['args'], method_config=config['coreset_method']['args'])

# Select subset
subset_indices = selector.select()

with open(f'indices/{dataset_name}_{method_name}_{str(fraction)}_main.pkl', 'wb') as f:
    pickle.dump(subset_indices, f)

# nohup python main.py > logs/main.log 2>&1 &
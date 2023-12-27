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
dataset_name = config['dataset']['name']
dataset = load_dataset(dataset_name)
if 'split' in config['dataset']:
    dataset = dataset[config['dataset']['split']]
    dataset = dataset.shuffle()

# Initialize selector according to yml
method_name = config['coreset_method']['name']
fraction = config['coreset_method']['args']['fraction']
selector = methods.__dict__[method_name](dataset, dataset_config=config['dataset']['args'], method_config=config['coreset_method']['args'])

# Select subset
subset_indices = selector.select()

with open(f'indices/{method_name}_{str(fraction)}.pkl', 'wb') as f:
    pickle.dump(subset_indices, f)
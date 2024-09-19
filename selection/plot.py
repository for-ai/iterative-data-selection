import argparse
import pickle
import datasets
from datasets import load_dataset
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DATASET_PATH = {
    'p3': 'simonycl/p3_0.5_dataset'
}
METHODS = ['Uniform', 'Random', 'kmeansrandom', 'KMeansCentroids']

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='p3')
    args.add_argument('--indices_path',
                      type=str,
                      default='indices')
    args.add_argument('--portion', type=float, default=0.1)
    args.add_argument('--output_path', type=str, default='plots')

    return args.parse_args()

def main():
    args = parse_args()
    print(args)
    # with open(args.indices_path, 'rb') as f:
    #     indices = pickle.load(f)

    dataset_path = DATASET_PATH[args.dataset]
    if dataset_path.endswith('json') or dataset_path.endswith('jsonl'):
        dataset = load_dataset('json', data_files=dataset_path)
    elif dataset_path.endswith('csv'):
        dataset = load_dataset('csv', data_files=dataset_path)
    else:
        dataset = load_dataset(dataset_path)

    dataset = dataset['train']

    method2indices = {}
    for method in METHODS:
        if os.path.exists(os.path.join(args.indices_path, f'{args.dataset}_{method}_{args.portion}.pkl')):
            with open(os.path.join(args.indices_path, f'{args.dataset}_{method}_{args.portion}.pkl'), 'rb') as f:
                method2indices[method] = pickle.load(f)
        else:
            print(f'{args.dataset}_{method}_{args.portion}.pkl does not exist')
    method2indices['full'] = list(range(len(dataset)))

    method2counters = {}
    for method, indices in method2indices.items():
        if isinstance(indices, dict):
            indices = indices['indices']
        subset = dataset.select(indices)
        # get the distribution of subset['category']
        counter = dict(Counter(subset['dataset']))
        # plot the distribution
        method2counters[method] = counter
        # counter['method'] = method
        # method2counters.append(counter)

    counters = []
    for method, counter in method2counters.items():
        if method == 'full':
            continue
        for key in counter:
            counter[key] /= method2counters['full'][key]
        counter['method'] = method
        counters.append(counter)

    # plot the distribution
    df = pd.DataFrame(counters)
    df = df.melt(id_vars=['method'], var_name='dataset', value_name='count')
    sns.barplot(x='dataset', y='count', hue='method', data=df)
    plt.savefig(os.path.join(args.output_path, f'{args.dataset}_{args.portion}.png'))

if __name__ == '__main__':
    main()
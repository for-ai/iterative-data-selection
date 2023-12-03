"""
Uniform is not a top-k ranking method.
"""
import numpy as np
from .coresetmethod import CoresetMethod
import warnings
import tqdm

def quasi_uniform_sampling(label_counts, num_samples):
    adjusted_samples = {label: 0 for label in label_counts}
    remaining_labels = set(label_counts.keys())

    pbar = tqdm.tqdm(total=num_samples)
    while num_samples > 0 and remaining_labels:
        evenly_distributed_samples = num_samples // len(remaining_labels)
        if evenly_distributed_samples == 0:
            break
        
        for label in list(remaining_labels):
            max_samples_for_label = min(evenly_distributed_samples, label_counts[label] - adjusted_samples[label])
            adjusted_samples[label] += max_samples_for_label
            num_samples -= max_samples_for_label
            pbar.update(max_samples_for_label)

            if adjusted_samples[label] == label_counts[label]:
                remaining_labels.remove(label)

    return adjusted_samples

class Uniform(CoresetMethod):
    def __init__(self, dataset, dataset_config, method_config):
        super().__init__(dataset, dataset_config, method_config)
        self._is_raking = False
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def select(self):
        label_column = self.dataset_config['label_column']
        labels = np.array(self.dataset[label_column])

        unique_labels = np.unique(labels)
        label_counts = {label: (labels == label).sum() for label in unique_labels}

        # Initial allocation
        samples_per_label = quasi_uniform_sampling(label_counts, self.coreset_size)

        # Check for underrepresented labels and issue warnings
        selected_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            selected_indices.extend(np.random.choice(label_indices, samples_per_label[label], replace=False))

        # Shuffle the selected indices to mix the labels
        np.random.shuffle(selected_indices)
        return {'indices': selected_indices}

    
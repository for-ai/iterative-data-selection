"""
Uniform is not a top-k ranking method.
"""
import numpy as np
from .coresetmethod import CoresetMethod
import warnings
import tqdm

class Uniform_Upsample(CoresetMethod):
    def __init__(self, dataset, dataset_config, method_config):
        super().__init__(dataset, dataset_config, method_config)
        self._is_raking = False
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def select(self):
        label_column = self.dataset_config['label_column']
        labels = np.array(self.dataset[label_column])

        unique_labels = np.unique(labels)

        total_size = len(labels)
        average_samples_per_label = total_size // len(unique_labels)
        # Check for underrepresented labels and issue warnings
        selected_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            selected_indices.extend(np.random.choice(label_indices, average_samples_per_label, replace=True))

        # Shuffle the selected indices to mix the labels
        np.random.shuffle(selected_indices)
        return {'indices': selected_indices}

    
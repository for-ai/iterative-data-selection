"""
KMeansRandom not is a top-k ranking method.

This method uses K-means clustering to select random samples of the clusters.

"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .coresetmethod import CoresetMethod
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random

class KMeansRandom(CoresetMethod):
    def __init__(self, dataset, dataset_config, method_config, K=20):
        super().__init__(dataset, dataset_config, method_config)
        self.K = K
        self._is_raking = False

    def select(self):
        data = self._extract_data()

        print("Performing K-means clustering...")
        # Perform K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=self.K, batch_size=128, random_state=self.random_seed, n_init="auto", verbose=1)
        kmeans.fit(data)

        # Assign data points to clusters
        labels = kmeans.labels_

        # Select random samples from each cluster
        selected_indices = self._select_random_samples(data, labels)
        return {'indices': selected_indices}

    def _extract_data(self):
        # Extract the data as a NumPy array from the dataset
        # This method needs to be implemented according to the format of your dataset
        sentences = self.dataset[self.dataset_config['data_column']]
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embeddings = model.encode(sentences, batch_size=512, device='cuda', show_progress_bar=True)
        return embeddings

    def _select_random_samples(self, data, labels):
        selected_indices = []
        n_samples = len(data)
        for cluster_id in range(self.K):
            cluster_indices = np.where(labels == cluster_id)[0]
            n_select = int(len(cluster_indices) / n_samples * self.coreset_size)
            selected_indices.extend(random.sample(list(cluster_indices), n_select))
        return selected_indices

"""
KMeansRandom not is a top-k ranking method.

This method uses K-means clustering to select random samples of the clusters.

"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .coresetmethod import CoresetMethod
import sys
sys.path.append('../')
from encoder import AutoEncoder
from tqdm import tqdm
import random
from encoder.utils import concat_tulu_messages, concat_tulu_messages_only_user, get_default_conv_template


class KMeansRandom(CoresetMethod):
    def __init__(self, dataset, data_config, method_config, encoder_config=None, K=1024):
        super().__init__(dataset, data_config, method_config, encoder_config)
        self.K = K
        self._is_raking = False
        self.encoder = AutoEncoder(encoder_config)

    def select(self):
        embeddings = self.encoder.get_embeddings(self.dataset, self.data_config)

        print("Performing K-means clustering...")
        # Perform K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=self.K, batch_size=128, random_state=self.random_seed, n_init="auto", verbose=1)
        kmeans.fit(embeddings)

        # Assign data points to clusters
        labels = kmeans.labels_

        # Select random samples from each cluster
        selected_indices = self._select_random_samples(embeddings, labels)
        return {'indices': selected_indices}

    def _select_random_samples(self, data, labels):
        selected_indices = []
        n_samples = len(data)
        for cluster_id in range(self.K):
            cluster_indices = np.where(labels == cluster_id)[0]
            n_select = int(len(cluster_indices) / n_samples * self.coreset_size)
            selected_indices.extend(random.sample(list(cluster_indices), n_select))
        return selected_indices

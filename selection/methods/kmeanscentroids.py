"""
KMeansCentroids is a top-k ranking method.

This method uses K-means clustering to select the centroids of the clusters. Set K = size of subset.

"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from .coresetmethod import CoresetMethod
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from encoder import AutoEncoder

class KMeansCentroids(CoresetMethod):
    def __init__(self, dataset, data_config, method_config, encoder_config=None):
        super().__init__(dataset, data_config, method_config, encoder_config)
        self.encoder = AutoEncoder(encoder_config)
        self._is_raking = False

    def select(self):
        embeddings = self.encoder.get_embeddings(self.dataset, self.data_config)

        print("Performing K-means clustering...")
        # Perform K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=self.coreset_size, batch_size=512, random_state=self.random_seed, n_init="auto", verbose=1)
        kmeans.fit(embeddings)

        # Select centroids
        centroids = kmeans.cluster_centers_
        print("Finding closest data points to centroids...")

        # Find the closest data points to centroids
        selected_indices = self._find_closest_data_points(embeddings, centroids)
        return {'indices': selected_indices}

    def _find_closest_data_points(self, data, centroids):
        closest_indices = []
        for centroid in tqdm(centroids):
            distances = np.linalg.norm(data - centroid, axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)
        return closest_indices
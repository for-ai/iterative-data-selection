"""
KMeansCentroids is a top-k ranking method.

This method uses K-means clustering to select the centroids of the clusters. Set K = size of subset.

"""

import numpy as np
from sklearn.cluster import KMeans
from .coresetmethod import CoresetMethod
from sentence_transformers import SentenceTransformer

class KMeansCentroids(CoresetMethod):
    def __init__(self, dataset, dataset_config, method_config):
        super().__init__(dataset, dataset_config, method_config)
        self._is_raking = False

    def select(self):
        data = self._extract_data()

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.coreset_size, random_state=self.random_seed)
        kmeans.fit(data)

        # Select centroids
        centroids = kmeans.cluster_centers_

        # Find the closest data points to centroids
        selected_indices = self._find_closest_data_points(data, centroids)
        return {'indices': selected_indices}

    def _extract_data(self):
        # Extract the data as a NumPy array from the dataset
        # This method needs to be implemented according to the format of your dataset
        sentences = self.dataset[self.dataset_config['data_column']]
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embeddings = model.encode(sentences, batch_size=256, device='cuda', show_progress_bar=True)
        return embeddings

    def _find_closest_data_points(self, data, centroids):
        closest_indices = []
        for centroid in centroids:
            distances = np.linalg.norm(data - centroid, axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)
        return closest_indices
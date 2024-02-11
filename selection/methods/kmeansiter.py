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
import faiss

class KMeansIter(CoresetMethod):
    def __init__(self, dataset, data_config, method_config, encoder_config=None, K=64, rounds=4, iter_data=None):
        super().__init__(dataset, data_config, method_config, encoder_config)
        self.K = K
        self._is_raking = False
        self.encoder = AutoEncoder(encoder_config)
        self.rounds = rounds
        if iter_data is None:
            self.iter = 0
            self.select_size = [int(self.coreset_size / self.rounds / self.K) for _ in range(self.K)]

    def select(self):
        embeddings = self.encoder.get_embeddings(self.dataset, self.data_config)

        print("Performing K-means clustering...")
        # Perform K-means clustering
        d = embeddings.shape[1]

        kmeans = faiss.Kmeans(d, self.K, niter=300, verbose=True, nredo=5, gpu=True)
        kmeans.train(embeddings)

        # get which centroid each embedding belongs to
        distances, indices = kmeans.index.search(embeddings, 1)

        # flatten indices
        indices = indices.reshape(-1)

        final_indices = np.array([], dtype=np.int64)
        for i in range(self.K):
            indices_i = np.where(indices == i)[0]
            size = np.minimum(int(self.select_size[i]), len(indices_i))
            indices_i = np.random.choice(indices_i, size=size, replace=False)
            final_indices = np.concatenate((final_indices, indices_i))

        print(final_indices.shape)
        return {'indices': final_indices, 'iter': self.iter}

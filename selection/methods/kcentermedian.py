from .deita.scorefaiss import DeitaScoreFaiss
import numpy as np
import json
import faiss

class KCenterMedian(DeitaScoreFaiss):
    def __init__(self, dataset, dataset_config, method_config, K=1024):
        super().__init__(dataset, dataset_config, method_config)
        self.K = K
        self._is_raking = True
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def select(self):
        embeddings = self.get_embeddings()

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)

        kmeans = faiss.Kmeans(d, self.K, niter=150, verbose=True, nredo=5, gpu=True)
        kmeans.train(embeddings)

        # get which centroid each embedding belongs to
        distances, indices = kmeans.index.search(embeddings, 1)

        # flatten indices
        indices = indices.reshape(-1)
        distances = distances.reshape(-1)
        
        final_indices = np.array([], dtype=np.int64)
        for i in range(self.K):
            indices_i = np.where(indices == i)[0]
            size = np.minimum(int(self.coreset_size/self.K), len(indices_i))
            # select the first size # of elements from the median distance to the centroid
            median_distance = np.median(distances[indices_i])
            indices_i = indices_i[np.argsort(np.abs(distances[indices_i] - median_distance))][:size]
            
            final_indices = np.concatenate((final_indices, indices_i))

        print(final_indices.shape)
        return {'indices': final_indices}
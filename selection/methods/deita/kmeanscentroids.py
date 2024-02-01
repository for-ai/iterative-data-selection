from .scorefaiss import DeitaScoreFaiss
import numpy as np
import json
import faiss

class KMeansCentroidDeita(DeitaScoreFaiss):
    def __init__(self, dataset, data_config, method_config, encoder_config=None):
        super().__init__(dataset, data_config, method_config, encoder_config)
        self._is_raking = True
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def select(self):
        embeddings = self.encoder.get_embeddings(self.dataset, self.data_config)

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)

        kmeans = faiss.Kmeans(d, self.coreset_size, niter=200, verbose=True, nredo=5, gpu=True)
        kmeans.train(embeddings)

        centroids = kmeans.centroids
        distances, indices = index.search(centroids, 1)
        # flatten indices
        indices = indices.reshape(-1)
        print(indices.shape)

        return {'indices': indices}

from .scorefaiss import DeitaScoreFaiss
import numpy as np
import json
import faiss

class KMenasRandomDeita(DeitaScoreFaiss):
    def __init__(self, dataset, dataset_config, method_config, K=1024):
        super().__init__(dataset, dataset_config, method_config)
        self.K = K
        self._is_raking = True
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def select(self):
        embeddings = self.get_embeddings()
        evol_scores, evol_ranking = self.get_scores()

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)

        kmeans = faiss.Kmeans(d, self.K, niter=200, verbose=True, nredo=5, gpu=True)
        kmeans.train(embeddings)

        # get which centroid each embedding belongs to
        distances, indices = kmeans.index.search(embeddings, 1)

        # flatten indices
        indices = indices.reshape(-1)

        final_indices = np.array([], dtype=np.int64)
        for i in range(self.K):
            indices_i = np.where(indices == i)[0]
            scores_i = evol_scores[indices_i]
            
            # select by random with probability proportional to score
            # set those scores < 0 to 0
            scores_i = np.maximum(scores_i, 0)
            p = scores_i / np.sum(scores_i)

            # handle the cases where some cluster has less than coreset_size/K elements
            size = np.minimum(int(self.coreset_size/self.K), len(indices_i))
            indices_i = np.random.choice(indices_i, size=size, replace=False, p=p)
            final_indices = np.concatenate((final_indices, indices_i))

        print(final_indices.shape)
        return {'indices': final_indices}
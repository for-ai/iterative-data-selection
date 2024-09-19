from .scorefaiss import DeitaScoreFaiss
import numpy as np
import json
import faiss

def safe_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class KMeansDynamic(DeitaScoreFaiss):
    def __init__(self, dataset, data_config, method_config, encoder_config=None, K=64):
        super().__init__(dataset, data_config, method_config, encoder_config)
        self.K = K
        self._is_raking = True
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.initial_seed = 10

    # def select_from_cluster(self, embeddings, indices, scores, )
    def select(self):
        embeddings = self.encoder.get_embeddings(self.dataset, self.data_config)
        evol_scores, evol_ranking = self.get_scores() # scores.shape = (n, 1)

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)

        kmeans = faiss.Kmeans(d, self.K, niter=300, verbose=True, nredo=5, gpu=True)
        kmeans.train(embeddings)

        # get which centroid each embedding belongs to
        distances, indices = kmeans.index.search(embeddings, 1)

        # flatten indices
        indices = indices.reshape(-1) # indices.shape = (n,1)

        final_indices = np.array([], dtype=np.int64)
        self.clusters2indices = {i : np.where(indices == i)[0] for i in range(self.K)}
        clusters2weights = np.ones(self.K) / self.K

        trial = np.ceil(self.coreset_size / (self.K * self.initial_seed)).astype(int)
        final_indices = np.array([], dtype=np.int64)
        for i in range(trial):
            select_sizes = np.ceil(clusters2weights * (self.initial_seed * self.K)).astype(int)

            # print('Select sizes:', select_sizes)
            for j in range(self.K):
                if (select_sizes[j] == 0) or (len(self.clusters2indices[j]) == 0):
                    clusters2weights[j] = 0
                    continue
                indices_j = self.clusters2indices[j]
                scores_j = evol_scores[indices_j]
                p = safe_softmax(scores_j)
                size = np.minimum(select_sizes[j], len(indices_j))
                indices_j = np.random.choice(indices_j, size=size, replace=False, p=p)
                final_indices = np.concatenate((final_indices, indices_j))
                self.clusters2indices[j] = np.setdiff1d(self.clusters2indices[j], indices_j)
                clusters2weights[j] = np.sum(evol_scores[indices_j])
            # perform average on clusters2weights
            clusters2weights = clusters2weights / np.sum(clusters2weights)
            # clusters2weights = safe_softmax(clusters2weights)

            # # perform softmax on clusters2weights
            # clusters2weights = np.exp(clusters2weights - np.max(clusters2weights))
            # clusters2weights = clusters2weights / np.sum(np.exp(clusters2weights - np.max(clusters2weights)))

        print(final_indices.shape)
        return {'indices': final_indices}

# CUDA_VISIBLE_DEVICES=1 python3 selection/main.py coreset=KMeansDynamic data=cohere encoder=llama
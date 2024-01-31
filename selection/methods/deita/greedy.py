from .scorefaiss import DeitaScoreFaiss
import numpy as np
import json
import faiss
from tqdm import tqdm

class KCenterGreedyDeita(DeitaScoreFaiss):
    def __init__(self, dataset, data_config, method_config, encoder_config=None):
        super().__init__(dataset, data_config, method_config, encoder_config)
        self._is_raking = True
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.threshold = method_config.get('threshold', 0.9)

    def select(self):
        embeddings = self.get_embeddings()

        evol_scores, evol_ranking = self.get_scores()

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)

        indices = []
        pbar = tqdm(total=self.coreset_size, desc="Greedy Selection")
        for idx in evol_ranking:
            if len(indices) >= self.coreset_size:
                break
            elif len(indices) == 0:
                indices.append(idx)
                index.add(embeddings[idx:idx+1])
                pbar.update(1)
                continue
            else:
                distances, _ = index.search(embeddings[idx:idx+1], 1)
                if distances[0][0] < self.threshold:
                    continue
                else:
                    indices.append(idx)
                    index.add(embeddings[idx:idx+1])
                    pbar.update(1)

        return {'indices': indices}

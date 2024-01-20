from ..coresetmethod import CoresetMethod
import numpy as np
import json
import sys
sys.path.append('../../')
from encoder import AutoEncoder
import faiss

class DeitaGreedy(CoresetMethod):
    def __init__(self, dataset, dataset_config, method_config):
        super().__init__(dataset, dataset_config, method_config)
        self._is_raking = True
        self._scores_path = method_config.get('scores_path', None)
        assert self._scores_path is not None, "scores_path must be specified in the config"
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.encoder = AutoEncoder(self.method_config['encoder_config'])
        self.batch_size = self.method_config['encoder_config']['batch_size'] if 'batch_size' in self.method_config['encoder_config'] else 256
        self.faiss = faiss.IndexFlatL2(self.encoder.encoder.model.hidden_size)
        
    def _greedy_select(self, scores, coreset_size, threshold=0.9):
        indices = []
        idx = 0
        while (len(indices) < coreset_size) and (idx < len(scores)):
            top_most_indices = scores[idx:idx+self.batch_size]
            # TODO how do you deal with the case when input data is multi-round dialogues?
            sentences = [self.dataset[idx] for idx in top_most_indices]
            embeddings = self.encoder.encode(sentences, batch_size=self.batch_size)
            embeddings = embeddings.astype('float32')
            distances, _indices = self.faiss.search(embeddings, 1)

            for i, (distance, index) in enumerate(zip(distances, _indices)):
                if distance[0] > threshold:
                    indices.append(top_most_indices[i])
                    self.faiss.add(embeddings[i])
            idx += self.batch_size

        return indices
    
    def select(self):
        with open(self._scores_path, "r") as f:
            lines = f.readlines()
        scores = [json.loads(line)['eval_score'] for line in lines]
        indices = self._greedy_select(scores, self.coreset_size)

        return {'indices': indices}
    
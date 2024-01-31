from ..coresetmethod import CoresetMethod
import numpy as np
import json

class DeitaScoreFaiss(CoresetMethod):
    def __init__(self, dataset, data_config, method_config, encoder_config=None):
        super().__init__(dataset, data_config, method_config, encoder_config)
        self._scores_path = data_config.get('scores_path', None)
        assert self._scores_path is not None, "scores_path must be specified in the config"
        self.embedding_cache_path = data_config.get('embedding_cache_path', None)
        assert self.embedding_cache_path is not None, "embedding_cache_path must be specified in the config"
        self.encoder_config = encoder_config
        if self.encoder_config:
            encoder_name = self.encoder_config.model_name
            if '/' in encoder_name:
                encoder_name = encoder_name.split('/')[1]
            self.encoder_name = encoder_name

    def get_embeddings(self):
        if (self.embedding_cache_path is not None) and (self.encoder_name in self.embedding_cache_path):
            try:
                print("Loading embeddings from cache...")
                embeddings = np.load(self.embedding_cache_path)
                print("Loaded embeddings from cache.")
            except FileNotFoundError:
                raise Exception("Embedding cache not found. Extracting embeddings from dataset...")
        else:
            raise NotImplementedError("Embedding cache not found. Extracting embeddings from dataset...")
        
        return embeddings

    def get_scores(self):
        with open(self._scores_path, "r") as f:
            lines = f.readlines()
        scores = [json.loads(line)['eval_score'] for line in lines]
        scores = np.array(scores, dtype=np.int64)
        ranking = np.argsort(scores)[::-1]
        return scores, ranking
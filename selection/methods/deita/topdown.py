from ..coresetmethod import CoresetMethod
import numpy as np
import json

class DeitaTopDown(CoresetMethod):
    def __init__(self, dataset, data_config, method_config):
        super().__init__(dataset, data_config, method_config)
        self._is_raking = True
        self._scores_path = method_config.get('scores_path', None)
        assert self._scores_path is not None, "scores_path must be specified in the config"
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def select(self):
        with open(self._scores_path, "r") as f:
            lines = f.readlines()
        scores = [json.loads(line)['eval_score'] for line in lines]
        ranking = np.argsort(scores)[::-1]
        return {'ranking': ranking, 'indices': ranking[:self.coreset_size]}
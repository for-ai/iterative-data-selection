from ..coresetmethod import CoresetMethod
import numpy as np
import json
from encoder import AutoEncoder

class DeitaScoreFaiss(CoresetMethod):
    def __init__(self, dataset, data_config, method_config, encoder_config=None):
        super().__init__(dataset, data_config, method_config, encoder_config)
        self._scores_path = data_config.get('scores_path', None)
        assert self._scores_path is not None, "scores_path must be specified in the config"
        self.encoder = AutoEncoder(encoder_config)
        self.encoder_config = encoder_config
        if self.encoder_config:
            encoder_name = self.encoder_config.model_name
            if '/' in encoder_name:
                encoder_name = encoder_name.split('/')[1]
            self.encoder_name = encoder_name

    def get_scores(self):
        with open(self._scores_path, "r") as f:
            lines = f.readlines()
        scores = [json.loads(line)['eval_score'] for line in lines]
        scores = np.array(scores, dtype=np.int64)
        ranking = np.argsort(scores)[::-1]
        return scores, ranking
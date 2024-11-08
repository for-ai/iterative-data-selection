"""
Random is a top-k ranking method.
"""

import numpy as np
from .coresetmethod import CoresetMethod

class Random(CoresetMethod):
    def __init__(self, dataset, data_config, method_config, encoder_config=None):
        super().__init__(dataset, data_config, method_config, encoder_config)
        self._is_raking = True
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def select(self):
        # Ensure unique selection of indices
        ranking = np.random.permutation(np.arange(len(self.dataset)))
        return {'ranking': ranking, 'indices': ranking[:self.coreset_size]}
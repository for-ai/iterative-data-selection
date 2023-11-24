class CoresetMethod(object):
    def __init__(self, dataset, dataset_config, method_config):
        fraction = method_config.get('fraction', 0.5)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")
        self.dataset = dataset
        self.fraction = fraction
        self.random_seed = method_config.get('random_seed', None)
        self.index = []
        self.dataset_config = dataset_config
        self.method_config = method_config
        self._is_raking = False

        self.n_dataset = len(dataset)
        self.coreset_size = round(self.n_dataset * fraction)

    def select(self, **kwargs):
        pass
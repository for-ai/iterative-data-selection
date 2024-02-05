import abc
from tqdm import tqdm
import numpy as np
from selection.encoder.utils import concat_messages

# Create an encoder class which is abstract
class Encoder(abc.ABC):
    def __init__(self, config):
        self.config = config
        
    @abc.abstractmethod
    def encode(self, example):
        pass

    def get_embeddings(self, dataset, data_config):
        # First check if cache is available
        is_encoder_cached = self.config.get('is_cached', False)
        concat_method = self.config.get('concat_method', 'tulu')
        if is_encoder_cached:
            encoder_name = self.config['model_name'].split('/')[-1]
            cache_path = '/'.join(data_config.name.split('/')[:-1] + [encoder_name + '-' + concat_method + '-embeddings.npy'])
            # cache_path = '/mnt/data/data-selection/data/processed/sharegpt/Llama-2-7b-hf-embeddings.npy'
            try:
                embeddings = np.load(cache_path)
                return embeddings
            except:
                pass
        # If not, extract the embeddings
        sentences = dataset[data_config.data_column]
        sentences = [concat_messages(message, concat_method) for message in tqdm(sentences, desc="Concatenating messages")]
        embeddings = self.encode(sentences, batch_size=self.config.batch_size, device='cuda', show_progress_bar=True)
        # Cache the embeddings
        if is_encoder_cached:
            np.save(cache_path, embeddings)
        return embeddings
"""
KMeansRandom not is a top-k ranking method.

This method uses K-means clustering to select random samples of the clusters.

"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .coresetmethod import CoresetMethod
import sys
sys.path.append('../')
from encoder import AutoEncoder
from tqdm import tqdm
import random

    
def _concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

class KMeansRandom(CoresetMethod):
    def __init__(self, dataset, dataset_config, method_config, K=1024):
        super().__init__(dataset, dataset_config, method_config)
        self.K = K
        self._is_raking = False
        self.embedding_cache_path = method_config.get('embedding_cache_path', None)
        if self.embedding_cache_path is None:
            self.embedding_cache_path = "embeddings.npy"

    def select(self):
        if self.embedding_cache_path is not None:
            try:
                print("Loading embeddings from cache...")
                data = np.load(self.embedding_cache_path)
                print("Loaded embeddings from cache.")
            except FileNotFoundError:
                print("Embedding cache not found. Extracting embeddings from dataset...")
                data = self._extract_data()
                print("Saving embeddings to cache...")
                np.save(self.embedding_cache_path, data)
                print("Saved embeddings to cache.")

        print("Performing K-means clustering...")
        # Perform K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=self.K, batch_size=128, random_state=self.random_seed, n_init="auto", verbose=1)
        kmeans.fit(data)

        # Assign data points to clusters
        labels = kmeans.labels_

        # Select random samples from each cluster
        selected_indices = self._select_random_samples(data, labels)
        return {'indices': selected_indices}

    def _extract_data(self):
        # Extract the data as a NumPy array from the dataset
        # This method needs to be implemented according to the format of your dataset
        sentences = self.dataset[self.dataset_config['data_column']]
        model = AutoEncoder(self.method_config['encoder_config'])
        # check if sentences is string or list
        if not isinstance(sentences[0], str):
            sentences = [_concat_messages(sentence, model.encoder.tokenizer) for sentence in tqdm(sentences, desc="Concatenating messages")]
        embeddings = model.encode(sentences, batch_size=64, device='cuda', show_progress_bar=True)
        return embeddings

    def _select_random_samples(self, data, labels):
        selected_indices = []
        n_samples = len(data)
        for cluster_id in range(self.K):
            cluster_indices = np.where(labels == cluster_id)[0]
            n_select = int(len(cluster_indices) / n_samples * self.coreset_size)
            selected_indices.extend(random.sample(list(cluster_indices), n_select))
        return selected_indices

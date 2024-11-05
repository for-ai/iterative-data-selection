from selection.encoder import Encoder
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, RateLimitError
import backoff
import os
import time

import tiktoken

def truncate_text_tokens(text, encoding_name="cl100k_base", max_tokens=2048):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    if text == '':
        return " "
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text,
                             allowed_special={'<|endoftext|>'},
                             )[:max_tokens]
    return encoding.decode(tokens)

class OpenAIEncoder(Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI()
        self.model_name = config['model_name']
        self.hidden_size = 1536
    
    @backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
    def embeddings_with_backoff(self, input, model):
        return self.client.embeddings.create(
            input = input,
            model = model,
        )
    
    def encode(self, sentences, batch_size=1024, device='cuda', show_progress_bar=True, aggregate_method='mean'):

        sentences = [truncate_text_tokens(sentence) for sentence in tqdm(sentences)]
        # check if there's any none or empty string
        any_none = any([s is None for s in sentences])
        any_empty = any([s == '' for s in sentences])
        if any_none:
            raise ValueError("Some sentences are None")
        if any_empty:
            raise ValueError("Some sentences are empty")
        
        embeddings = np.zeros((len(sentences), self.hidden_size))

        for i in trange(0, len(sentences), batch_size):
            batch_instances = sentences[i:i+batch_size]

            response = self.embeddings_with_backoff(batch_instances, self.model_name)

            for j in range(min(batch_size, len(sentences) - i)):
                embeddings[i+j] = response.data[j].embedding
            # add a sleep to avoid rate limit
            # time.sleep(0.5)

        # print the device where the embeddings are stored
        # print("Embeddings are stored in: ", embeddings.device)
        return embeddings

# sanity check
# if __name__ == '__main__':
#     # encoder = SemanticBasedEncoder({'model_name': 'intfloat/e5-large-v2', 'max_length_threshold': 2})
#     encoder = CohereEncoder({'model_name': 'embed-multilingual-v3.0'})
#     print(len(encoder._chunk_sentence('hello ' * 513)))
#     embeddings = encoder.encode(['hello ' * 513] * 1)
#     print(embeddings.shape)
#     print(encoder.encode(['hello ' * 513] * 1))
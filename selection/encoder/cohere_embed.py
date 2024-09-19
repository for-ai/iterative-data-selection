from selection.encoder import Encoder
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor
import cohere

class CohereEncoder(Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.co = cohere.Client("nNjVPgvGPgpCePFCcXDmhmuYFwYnKR7vlMAgofo9")
        self.model_name = config['model_name']
        self.max_seq_length = 512
        self.hidden_size = 1024
        # if self.config['is_sentence_transformer']:
        #     self.model = SentenceTransformer(self.config['model_name'], cache_folder='/mnt/data/.cache')
        #     self.tokenizer = self.model.tokenizer
        #     self.max_seq_length = self.model.max_seq_length if self.config.get('max_seq_length', None) is None else self.config['max_seq_length']
        #     self.hidden_size = self.model.get_sentence_embedding_dimension()
        # else:
        #     self.model = AutoModel.from_pretrained(self.config['model_name']).to('cuda')
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        #     self.max_seq_length = self.model.config.max_position_embeddings - 2 if self.config.get('max_seq_length', None) is None else self.config['max_seq_length']
        #     self.hidden_size = self.model.config.hidden_size
        # if 'max_length_threshold' in self.config:
        #     self.max_seq_length -= self.config['max_length_threshold']

    def _chunk_sentence(self, sentence):
        '''
        Chunk the sentence into multiple sentences with the max length
        '''
        chunks = []
        start_idx = 0
        tokenized_sentence = sentence.split()
        while start_idx < len(tokenized_sentence):
            end_idx = start_idx + self.max_seq_length
            chunks.append(' '.join(tokenized_sentence[start_idx:end_idx]))
            start_idx = end_idx
        if len(chunks) == 0:
            chunks.append(sentence)
        return chunks
    
    def encode(self, sentences, batch_size=1024, device='cuda', show_progress_bar=True, aggregate_method='mean'):
        assert aggregate_method in ['mean', 'sum', 'max'], 'aggregate_method must be one of mean, sum, max'
        # chunk the sentences if they are longer than the max length
        chunked_instances = []
        chunked_indices = []
        def process_sentence(index, sentence):
            # Assuming self._chunk_sentence is a method that processes a sentence
            # Replace with the actual method from your context
            chunked = self._chunk_sentence(sentence)
            indices = [index] * len(chunked)
            return chunked, indices

        # Assuming 'sentences' is your list of sentences
        with ThreadPoolExecutor() as executor:
            # Map the processing function to each sentence
            results = list(tqdm(executor.map(lambda p: process_sentence(*p), enumerate(sentences)), total=len(sentences)))

        # wait for the results
        results = [r for r in results]

        # Unpack the results
        chunked_instances, chunked_indices = zip(*results)

        # Flatten the lists
        chunked_instances = [item for sublist in chunked_instances for item in sublist]
        chunked_indices = [item for sublist in chunked_indices for item in sublist]

        chunked_embeddings = np.zeros((len(chunked_instances), self.hidden_size))
        for i in trange(0, len(chunked_instances), batch_size):
            batch_instances = chunked_instances[i:i+batch_size]
            response = self.co.embed(
                texts = batch_instances,
                model = "embed-multilingual-v3.0",
                input_type = "clustering",
                truncate="END",
            )
            batch_embeddings = response.embeddings
            # batch_embeddings = response["embeddings"]
            chunked_embeddings[i:i+batch_size] = batch_embeddings

        # process the remaining instances
        remaining_instances = chunked_instances[i+batch_size:]
        response = self.co.embed(
            texts = remaining_instances,
            model = "embed-multilingual-v3.0",
            input_type = "clustering",
            truncate="END",
        )
        remaining_embeddings = response.embeddings
        chunked_embeddings[i+batch_size:] = remaining_embeddings

        # aggregate the embeddings
        embeddings = np.zeros((len(sentences), self.hidden_size))
        for i, index in enumerate(chunked_indices):
            if aggregate_method == 'mean':
                embeddings[index] += chunked_embeddings[i] / len(chunked_instances[index])
            elif aggregate_method == 'sum':
                embeddings[index] += chunked_embeddings[i]
            elif aggregate_method == 'max':
                embeddings[index] = np.maximum(embeddings[index], chunked_embeddings[i])
        
        print(embeddings.shape)
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
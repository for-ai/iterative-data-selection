from selection.encoder import Encoder
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor

class SemanticBasedEncoder(Encoder):
    def __init__(self, config):
        super().__init__(config)
        if self.config['is_sentence_transformer']:
            self.model = SentenceTransformer(self.config['model_name'], cache_folder='/mnt/data/.cache')
            self.tokenizer = self.model.tokenizer
            self.max_seq_length = self.model.max_seq_length if self.config.get('max_seq_length', None) is None else self.config['max_seq_length']
            self.hidden_size = self.model.get_sentence_embedding_dimension()
        else:
            self.model = AutoModel.from_pretrained(self.config['model_name']).to('cuda')
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
            self.max_seq_length = self.model.config.max_position_embeddings - 2 if self.config.get('max_seq_length', None) is None else self.config['max_seq_length']
            self.hidden_size = self.model.config.hidden_size
        if 'max_length_threshold' in self.config:
            self.max_seq_length -= self.config['max_length_threshold']

    def _chunk_sentence(self, sentence):
        '''
        Chunk the sentence into multiple sentences with the max length
        '''
        chunks = []
        start_idx = 0
        tokenized_sentence = self.tokenizer(sentence)['input_ids']
        tokenized_sentence = tokenized_sentence[1:-1]
        while start_idx < len(tokenized_sentence):
            end_idx = start_idx + self.max_seq_length
            chunks.append(self.tokenizer.decode(tokenized_sentence[start_idx:end_idx]))
            start_idx = end_idx
        return chunks
    
    def _preprocess_e5(self, sentences):
        '''
        Preprocess the sentences for the E5 model
        "query: " take two tokens, so the max length is 510
        '''
        # prepend the tokens for querying
        prepended_sentences = []
        for sentence in sentences:
            prepended_sentences.append(f'query: {sentence}')
        return prepended_sentences
    def _preprocess_e5_mistral(self, sentences):
        '''
        Preprocess the sentences for the E5 model
        "query: " take two tokens, so the max length is 510
        '''
        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'
        # prepend the tokens for querying
        prepended_sentences = []
        for sentence in sentences:
            prepended_sentences.append(get_detailed_instruct('Identify the topic or theme of the given text.', sentence))
        return prepended_sentences
    
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

        # Unpack the results
        chunked_instances, chunked_indices = zip(*results)

        # Flatten the lists
        chunked_instances = [item for sublist in chunked_instances for item in sublist]
        chunked_indices = [item for sublist in chunked_indices for item in sublist]
        
        # E5 model requires prepending the tokens for querying
        if 'e5-mistral' in self.config['model_name']:
            chunked_instances = self._preprocess_e5_mistral(chunked_instances)
        elif 'e5' in self.config['model_name']:
            chunked_instances = self._preprocess_e5(chunked_instances)

        chunked_embeddings = np.zeros((len(chunked_instances), self.hidden_size))
        if self.config['is_sentence_transformer']:
            chunked_embeddings = self.model.encode(chunked_instances, batch_size=batch_size, device=device, show_progress_bar=show_progress_bar)
        else:
            chunked_embeddings = np.zeros((len(chunked_instances), self.hidden_size))
            for i in trange(0, len(chunked_instances), batch_size):
                batch_instances = chunked_instances[i:i+batch_size]
                # (batch_size, embedding_size)
                batch_tokenized = self.tokenizer(batch_instances, return_tensors='pt', padding="longest")
                batch_embeddings = self.model(**batch_tokenized.to(device), output_hidden_states=True).hidden_states[-1][:, 0, :].detach().cpu().numpy()
                chunked_embeddings[i:i+batch_size] = batch_embeddings

        # aggregate the embeddings
        embeddings = np.zeros((len(sentences), self.hidden_size))
        for i, index in enumerate(chunked_indices):
            if aggregate_method == 'mean':
                embeddings[index] += chunked_embeddings[i] / len(chunked_instances[index])
            elif aggregate_method == 'sum':
                embeddings[index] += chunked_embeddings[i]
            elif aggregate_method == 'max':
                embeddings[index] = np.maximum(embeddings[index], chunked_embeddings[i])
        
        # print the device where the embeddings are stored
        # print("Embeddings are stored in: ", embeddings.device)
        return embeddings

# sanity check
# if __name__ == '__main__':
#     # encoder = SemanticBasedEncoder({'model_name': 'intfloat/e5-large-v2', 'max_length_threshold': 2})
#     encoder = SemanticBasedEncoder({'model_name': 'facebook/contriever', 
#                                     'is_sentence_transformer': False})
#     print(len(encoder._chunk_sentence('hello ' * 513)))
#     embeddings = encoder.encode(['hello ' * 513] * 1)
#     print(embeddings.shape)
    # print(encoder.encode(['hello ' * 513] * 1))
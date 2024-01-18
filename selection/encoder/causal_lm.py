from selection.encoder import Encoder
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm, trange
import numpy as np

class ModelBasedEncoder(Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = AutoModel.from_pretrained(
            self.config['model_name'],
            use_flash_attention_2=self.config['use_flash_attn'],
            torch_dtype=torch.bfloat16 if self.config['use_flash_attn'] else torch.float16,
            load_in_8bit=self.config['is_8bit'],
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.max_seq_length = self.config['max_seq_length'] if 'max_seq_length' in self.config else self.model.config.max_position_embeddings - 2
    
    @torch.no_grad()
    def encode(self, sentences, batch_size=256, device='cuda', show_progress_bar=True, aggregate_method='mean'):
        if 'batch_size' in self.config:
            batch_size = self.config['batch_size']
        embeddings = torch.zeros((len(sentences), self.model.config.hidden_size))
        for i in trange(0, len(sentences), batch_size):
            batch_instances = sentences[i:i+batch_size]
            tokenized_example = self.tokenizer(batch_instances, return_tensors='pt', padding="longest")
            batch_embeddings = self.model(**tokenized_example.to(device), output_hidden_states=True).hidden_states[-1][:, -1, :]
            embeddings[i:i+batch_size] = batch_embeddings
        return embeddings.detach().cpu().numpy()


# sanity check
# if __name__ == '__main__':
#     sentences = ['I love you', 'I hate you', 'what am I doing']
#     encoder = CasualModelEncoder({'model_name': 'meta-llama/Llama-2-7b-hf', 'use_flash_attn': True, 'is_8bit': False})
#     embeddings = encoder.encode(sentences)
#     print(embeddings)
#     print(np.all(np.isclose(embeddings[0], embeddings[1], atol=1e-3)))
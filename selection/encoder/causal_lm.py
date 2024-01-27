from selection.encoder import Encoder
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm, trange
import numpy as np
from torch import Tensor

def last_token_pool(last_hidden_states: Tensor,
                attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
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
        self.hidden_size = self.model.config.hidden_size
        
    @torch.no_grad()
    def encode(self, sentences, batch_size=256, device='cuda', show_progress_bar=True, aggregate_method='mean'):
        if 'batch_size' in self.config:
            batch_size = self.config['batch_size']
        self.model = self.model.to(device)
        embeddings = torch.zeros((len(sentences), self.hidden_size))
        for i in trange(0, len(sentences), batch_size):
            batch_instances = sentences[i:i+batch_size]
            tokenized_example = self.tokenizer(batch_instances, return_tensors='pt', padding="longest")
            batch_last_hidden_states = self.model(**tokenized_example.to(device), output_hidden_states=True).hidden_states[-1]
            embeddings[i:i+batch_size] = last_token_pool(batch_last_hidden_states, tokenized_example['attention_mask'])

        return embeddings.detach().cpu().numpy()


# # sanity check
# if __name__ == '__main__':
#     sentences = ['I love you', 'I hate you', 'what am I doing']
#     encoder = ModelBasedEncoder({
#         'model_name': 'intfloat/e5-mistral-7b-instruct',
#         'use_flash_attn': True, 
#         'is_8bit': False}
#         )
#     embeddings = encoder.encode(sentences)
#     print(embeddings)
#     print(np.all(np.isclose(embeddings[0], embeddings[1], atol=1e-3)))
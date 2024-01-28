from .base import Encoder
from .sentence_transformer import SemanticBasedEncoder
from .causal_lm import ModelBasedEncoder
from .utils import concat_tulu_messages, Conversation, SeparatorStyle, get_default_conv_template

class AutoEncoder(Encoder):
    def __init__(self, config):
        super().__init__(config)
        if 'is_sentence_transformer' in self.config:
            self.encoder = SemanticBasedEncoder(self.config)
        else:
            self.encoder = ModelBasedEncoder(self.config)

    def encode(self, sentences, batch_size=1024, device='cuda', show_progress_bar=True, aggregate_method='mean'):
        return self.encoder.encode(sentences, batch_size, device, show_progress_bar, aggregate_method)
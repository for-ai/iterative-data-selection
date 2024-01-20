import abc

import torch

# Create an encoder class which is abstract
class Encoder(abc.ABC):
    def __init__(self, config):
        self.config = config
        
    @abc.abstractmethod
    def encode(self, example):
        pass

import copy
import json

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class P3Dataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        if type(dataset_config.dataset) is tuple:
            self.dataset = load_dataset(dataset_config.dataset[0], dataset_config.dataset[1])
        
        if partition == "train":
            self.dataset = self.dataset[dataset_config.train_split]
        else:
            self.dataset = self.dataset[dataset_config.test_split]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        data = self.dataset[index]
        prompt = data['inputs_pretokenized']
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = prompt + data["targets_pretokenized"]
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }

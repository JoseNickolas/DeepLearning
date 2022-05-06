from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F
from mimic import Mimic3

from collections import Counter
import numpy as np

from tqdm import tqdm


class Word2Vec(nn.Module):
    def __init__(self, emb_dim, num_codes):
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=num_codes, embedding_dim=emb_dim)
        self.linear = nn.Linear(emb_dim, num_codes)

    def forward(self, x):
        x = self.emb(x)
        return self.linear(x)


class SkipGramDataset(Dataset): # TODO: use IterableDataset instead
    def __init__(self, dataset: Mimic3, alpha, beta):
        """
        Args:
            data (list[list[medical codes]]): list of medical concepts of patients.
        """
        super().__init__()
        
        self.dataset = dataset
        self.alpha = alpha
        self.beta = beta

        self.pairs = self.build_center_context_pairs()


    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

    def build_center_context_pairs(self):
        pairs = []
        for codes, _ in tqdm(self.dataset):
            counts = Counter(codes)
            for i, center in enumerate(codes):
                window = np.clip(counts[center], self.alpha, self.beta)
                contexts = codes[i - window: i] + codes[i + 1: i + 1 + window]
                for context in contexts:
                    pairs.append((center, context))
        return pairs


class SkipGramIterDataset(IterableDataset):
    def __init__(self, dataset: Mimic3, alpha, beta):
        """
        Args:
            data (list[list[medical codes]]): list of medical concepts of patients.
        """
        super().__init__()
        
        self.dataset = dataset
        self.alpha = alpha
        self.beta = beta
    
    def __len__(self):
        """ Estimate of number of (center,context) pairs"""
        total_ncodes = self.dataset.data.apply(len).sum()
        return int(total_ncodes * (self.alpha + self.beta) / 2)

    def __iter__(self):
        for codes, _ in self.dataset:
            counts = Counter(codes)
            for i, center in enumerate(codes):
                window = np.clip(counts[center], self.alpha, self.beta)
                contexts = codes[i - window: i] + codes[i + 1: i + 1 + window]
                for context in contexts:
                    yield (center, context)
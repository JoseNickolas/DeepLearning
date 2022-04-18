from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from siamese_cnn import SiameseCNN
from word2vec import Word2Vec


class PatientSimiEval(nn.Module):
    def __init__(self, word2vec, feature_maps, kernel_size, spp_levels, out_dim):
        super().__init__()
        self.embedding = word2vec
        self.scnn = SiameseCNN(feature_maps, kernel_size, spp_levels, out_dim)
        
        self.embedding.requires_grad_(False)
        
    def forward(self, code_seq):
        result = []
        for codes in code_seq: # TODO: use PackedSquence, or padding, move for loop to SPP.
            embs = self.embedding(codes)  # ncodes, emb_dim
            embs = embs[None, None, :, :] # 1 (batch), 1 (channel), ncodes, emb_dim
            pater_rep = self.scnn(embs)
            result.append(pater_rep)
        
        return torch.cat(result, dim=0) # batch, out_dim
        
        
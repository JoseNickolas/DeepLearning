from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .mimic import Mimic3
from .model import PatientSimiEval
from .siamese_cnn import SiameseCNN
from .utils import pairwise_intersect
from .word2vec import SkipGramDataset, Word2Vec

# constants
# DATASET_PATH = '../../../Project/Dataset/all_files/'
DATASET_PATH = '/content/drive/MyDrive/mimic3/'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Hyperparamers:
# word2vec
emb_dim = 100
alpha = 1
beta = 3
sg_lr = 0.1
sg_epochs = 1
sg_batch = 512
# siamese
feature_maps = 100
kernel_size = 5
spp_levels = (4, 2, 1)
out_dim = 10
scnn_epochs = 1000
scnn_batch = 32
sc_lr = 0.1
margin = 5


# create the main dataset of medical concept sequences (preprocessing) 
dataset = Mimic3(DATASET_PATH,
                 apply_vocab=True,
                 to_tensor=False)

num_codes = dataset.num_codes() # total number of unique medical codes
sg_dataset = SkipGramDataset(dataset, alpha=alpha, beta=beta)


def train_word2vec():
    # train word2vec model
    word2vec = Word2Vec(emb_dim, num_codes)
    word2vec.to(DEVICE)

    optim = torch.optim.Adam(lr=sg_lr, params=word2vec.parameters())

    sg_dataloader = DataLoader(sg_dataset, batch_size=sg_batch)
    
    for i in trange(sg_epochs, desc='Training word2vec'):
        for center, context in tqdm(sg_dataloader, desc=f'epoch {i}'):
            center, context = center.to(DEVICE), context.to(DEVICE)
            
            pred = word2vec(center)
            loss = F.cross_entropy(pred, context)
            
            loss.backward()
            optim.step()
            optim.zero_grad()
    
    return word2vec


def coallate(batch):
    codes, cohorts = zip(*batch)
    codes = [torch.tensor(c, dtype=torch.long, device=DEVICE) for c in codes]
    return codes, cohorts


def contrastive_loss(x, y): 
    pdist = 1 - F.cosine_similarity(x, x[:, None, :], dim=-1)
    
    y = pairwise_intersect(y)
    y = torch.tensor(y, device=DEVICE)
    y = 2 * y -1
    
    loss = F.hinge_embedding_loss(pdist, y) ** 2
    return loss

# train the model: pretrained word2vec + Siamese CNN with SPP


def train_pse(word2vec):
    pse_model = PatientSimiEval(word2vec.emb, feature_maps, kernel_size, spp_levels, out_dim)
    pse_model.to(DEVICE)


    data_loader = DataLoader(dataset, 
                            batch_size=scnn_batch,
                            collate_fn=coallate,
                            shuffle=True,
                            drop_last=True)

    params = (p for p in pse_model.parameters() if p.requires_grad)
    optim = torch.optim.Adam(lr=sc_lr, params=params)

    for epoch in trange(scnn_epochs, desc='Training PSE'):    
        for codes, cohorts in tqdm(data_loader):
            reps = pse_model(codes)
            
            loss = contrastive_loss(reps, cohorts)      
            loss.backward()
            
            optim.step()
            optim.zero_grad()
        
    return pse_model
        
        
   
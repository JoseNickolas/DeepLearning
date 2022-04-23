from tkinter import N
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from mimic import Mimic3
from model import PatientSimiEval
from siamese_cnn import SiameseCNN
from utils import pairwise_intersect
from word2vec import SkipGramDataset, Word2Vec
from datetime import datetime

# constants
DATASET_PATH = '../../../Project/Dataset/all_files/'
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


# train word2vec model
word2vec = Word2Vec(emb_dim, num_codes)
word2vec.to(DEVICE)

sg_dataset = SkipGramDataset(dataset, alpha=alpha, beta=beta)
sg_dataloader = DataLoader(sg_dataset, batch_size=sg_batch)

optim = torch.optim.Adam(lr=sg_lr, params=word2vec.parameters())

for i in trange(sg_epochs, desc='Training word2vec'):
    for center, context in tqdm(sg_dataloader, desc=f'epoch {i}'):
        center, context = center.to(DEVICE), context.to(DEVICE)
        
        pred = word2vec(center)
        loss = F.cross_entropy(pred, context)
        
        loss.backward()
        optim.step()
        optim.zero_grad()


# train the model: pretrained word2vec + Siamese CNN with SPP

pse_model = PatientSimiEval(word2vec.emb, feature_maps, kernel_size, spp_levels, out_dim)
pse_model.to(DEVICE)

def coallate(batch):
    codes, cohorts = zip(*batch)
    codes = [torch.tensor(c, dtype=torch.long, device=DEVICE) for c in codes]
    return codes, cohorts
    
data_loader = DataLoader(dataset, 
                         batch_size=scnn_batch,
                         collate_fn=coallate,
                         shuffle=True,
                         drop_last=True)

params = (p for p in pse_model.parameters() if p.requires_grad)
optim = torch.optim.Adam(lr=sc_lr, params=params)


def contrastive_loss(x, y): 
    pdist = 1 - F.cosine_similarity(x, x[:, None, :], dim=-1)
    
    y = pairwise_intersect(y)
    y = torch.tensor(y, device=DEVICE)
    y = 2 * y -1
    
    loss = F.hinge_embedding_loss(pdist, y) ** 2
    return loss


for epoch in trange(scnn_epochs, desc='Training PSE'):    
    for codes, cohorts in tqdm(data_loader):
        reps = pse_model(codes)
        
        loss = contrastive_loss(reps, cohorts)      
        loss.backward()
        
        optim.step()
        optim.zero_grad()
        
        
        
        

# TODO create validation set, and evalute


# most similar patients:
reprs = [pse_model(codes).detach().cpu().numpy() for codes, _ in data_loader]
reprs = np.vstack(reprs)

pdist =  cosine_similarity(reprs)
np.fill_diagonal(pdist, -2)
nearest_pse = pdist.argmax(axis=1)


# BASELINES
codes_onehot = np.zeros((len(dataset), dataset.num_codes()), dtype=np.int32)
for i, (codes, _) in enumerate(dataset):
    codes[i, codes] = 1
    
# PCA: pca of one-hode encoded codes
pca = PCA(n_components=out_dim) #  cohorts
codes_pca = pca.fit_transform(codes_onehot) # (N, 9)
pdist_pca = pairwise_distances(codes_pca) # pairwise euclidean dist
np.fill_diagonal(pdist_pca, np.inf) # ignore the diaginals
nearest_pca = pdist_pca.argmin(axis=1) # closest patients based on PCA

# Hamming distance:
pdist_hamming = pairwise_distances(codes_onehot, metirc='hamming')
np.fill_diagonal(pdist_hamming, np.inf) # ignore the diaginals
nearest_hamming = pdist_hamming.argmin(axis=1) # closest patients based on PCA



# The Hospital Readmission Rate (HRR)
nadmin = dataset.info['NADMISSIONS'].to_numpy()
def hospital_readmin_rate(nearest):
    return ((nadmin[nearest] == 1) & (nadmin == 1)) | ((nadmin[nearest] > 1) & (nadmin > 1)).mean()

# Incidence Rate Difference for Mortality (IRDM)
dischtime = dataset.info['DISCHTIME']
deathtime = dataset.info['DEATHTIME']

def incidence_rate(dischtime, deathtime):
    death_ct = (~deathtime.isna()).sum()
    diff = (deathtime.fillna(datetime(2022, 1, 1)) - dischtime).dt.days
    return death_ct / diff

def inc_rate_mortality(nearest):
    ir_p = incidence_rate(dischtime, deathtime)
    ir_sp = incidence_rate(dischtime.iloc[nearest], deathtime.iloc[nearest])
    return np.abs(ir_p - ir_sp)



methods = {
    "PSE": nearest_pse,
    "PCA": nearest_pca,
    "Hamming": nearest_hamming
}

irdm = {}
hrr = {}
for name, nearest in methods.items():
    irdm[name] = inc_rate_mortality(nearest)
    hrr[name] = hospital_readmin_rate(nearest)

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm, trange

from mimic import Mimic3, coallate
from model import PatientSimiEval
from utils import pairwise_intersect, save_config
from word2vec import SkipGramDataset, SkipGramIterDataset, Word2Vec

# constants
# DATASET_PATH = '../../../Project/Dataset/all_files/'
DATASET_PATH = '/content/drive/MyDrive/mimic3/'
MODEL_PATH = '/content/drive/MyDrive/mimic3/saved_models/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#######################
###### Word2Vec #######
#######################
    
def train_word2vec(config, mimic_ds, jit=False):
    """ train word2vec model """

    # create sg_dataset
    sg_dataset = SkipGramDataset(mimic_ds, 
                                 alpha=config['alpha'],
                                 beta=config['beta'])
    # sg_dataset = SkipGramIterDataset(dataset, config['alpha'], config['beta'])
    
    sg_dataloader = DataLoader(sg_dataset,
                               batch_size=config['sg_batch'],
                               shuffle=True)

    
    # logging, checkpointg
    suffix = 'w2v_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(MODEL_PATH) / 'logs' / suffix
    writer = SummaryWriter(log_dir)

    save_path = Path(MODEL_PATH) / suffix 
    save_path.mkdir()
    save_config(config, save_path)


    # instantiate model
    num_codes = mimic_ds.num_codes() # total number of unique medical codes
    word2vec = Word2Vec(config['emb_dim'], num_codes)
    word2vec.to(DEVICE)


    # jit
    if jit:
        center, _ = next(iter(sg_dataloader))
        center = center.to(DEVICE)
        word2vec = torch.jit.script(word2vec, example_inputs=(center,))
        save_fn = torch.jit.save
    else:
        save_fn = torch.save


    # train
    optim = torch.optim.Adam(lr=config['sg_lr'], params=word2vec.parameters())
    
    running_loss = 0
    for epoch in trange(config['sg_epochs'], desc='Training word2vec'):
        for i, (center, context) in enumerate(tqdm(sg_dataloader)):
            center, context = center.to(DEVICE), context.to(DEVICE)
            
            pred = word2vec(center)
            loss = F.cross_entropy(pred, context)
            
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            running_loss += loss.item()

            if i % 1000 == 999:
                # ...log the running loss
                writer.add_scalar('training loss',
                                  running_loss / 1000,
                                  epoch * len(sg_dataloader) + i)
                running_loss = 0

            if i % 100000 == 0:
                save_fn(word2vec, save_path / f'word2vec_e{epoch}_i{i}.pt' )

        save_fn(word2vec, save_path / f'word2vec_epoch_{epoch}.pt' )
    
    return word2vec



######################
#### Siamese CNN #####
######################

def contrastive_loss(x, y, margin): 
    y = pairwise_intersect(y)
    y = torch.tensor(y, device=DEVICE, dtype=torch.long)
    y = 2 * y -1
    
    pdist = 1 - F.cosine_similarity(x, x[:, None, :], dim=-1)
    loss = F.hinge_embedding_loss(pdist, y, margin=margin) ** 2
    return loss


def train_pse(word2vec, config, mimic_ds, jit=False):
    """ train a full PSE model: pretrained word2vec + Siamese CNN with SPP"""

    # logging, checkpointg
    suffix = 'pse_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(MODEL_PATH) / 'logs' / suffix
    writer = SummaryWriter(log_dir)

    save_path = Path(MODEL_PATH) / suffix 
    save_path.mkdir()
    save_config(config, save_path)

    # instantiate model
    pse_model = PatientSimiEval(word2vec.emb, 
                                config['feature_maps'], 
                                config['kernel_size'], 
                                config['spp_levels'],
                                config['out_dim'])
    pse_model.to(DEVICE)

    
    data_loader = DataLoader(mimic_ds, 
                            batch_size=config['scnn_batch'],
                            collate_fn=coallate,
                            shuffle=True,
                            drop_last=True)
    
    # jit
    if jit:
        codes, _ = next(iter(data_loader))
        pse_model = torch.jit.script(pse_model, example_inputs=(codes,))
        save_fn = torch.jit.save
    else:
        save_fn = torch.save


    # training loop
    params = (p for p in pse_model.parameters() if p.requires_grad)
    optim = torch.optim.Adam(lr=config['sc_lr'], params=params)

    running_loss = 0
    for epoch in trange(config['scnn_epochs'], desc='Training PSE'):    
        for i, (codes, cohorts) in enumerate(tqdm(data_loader)):
            reps = pse_model(codes)
            
            loss = contrastive_loss(reps, cohorts, margin=config['margin'])      
            loss.backward()
            
            optim.step()
            optim.zero_grad()
            
            running_loss += loss.item()

            if i % 20 == 19:
                # ...log the running loss
                writer.add_scalar('training loss',
                                  running_loss / 20,
                                  epoch * len(data_loader) + i)
                running_loss = 0
            
        save_fn(pse_model, save_path / f'pse_epoch{epoch}' )
        
    return pse_model



if __name__ == '__main__':

    # fix seeds
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


    # create the main dataset of medical concept sequences (preprocessing) 
    mimic_ds = Mimic3(DATASET_PATH,
                     apply_vocab=True,
                     to_tensor=False)


    # Hyperparameters
    config = {
        'word2vec': {
            'emb_dim': 100,
            'alpha': 20,
            'beta': 50,
            'sg_lr': 0.1,
            'sg_epochs': 500,
            'sg_batch': 128,
        },  
        
        'pse': {
            'feature_maps': 100,
            'kernel_size': 5,
            'spp_levels': (4, 2, 1),
            'out_dim': 10,
            'scnn_epochs': 500,
            'scnn_batch': 128,
            'sc_lr': 0.05,
            'margin': 1,
        },
    }

    # train a word2vec model
    # word2vec = train_word2vec(config['word2vec'], mimic_ds, jit=False)

    # or load a trained word2vec model
    w2v = {
        'name': 'w2v_20220503-220149',
        'version': 'word2vec_e19_i1100000.pt'
    }
    w2v_path = Path(MODEL_PATH) / w2v['name'] / w2v['version']
    word2vec = torch.load(w2v_path)


    # train the full model
    pse_model = train_pse(word2vec, config['pse'], mimic_ds, jit=False)

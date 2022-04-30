from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from mimic import Mimic3
from model import PatientSimiEval
from siamese_cnn import SiameseCNN
from utils import pairwise_intersect
from word2vec import SkipGramDataset, Word2Vec


# pse_model = torch.load('')


# most similar patients:
reprs_pse = [pse_model(codes).detach().cpu().numpy() for codes, _ in tqdm(data_loader)]
reprs_pse = np.vstack(reprs_pse)

def find_nearest(representation, metric):
    nearest = []
    for pdist in pairwise_distances_chunked(representation, metric=metric):
        if metric == 'cosine':
            pdist = 1 - pdist
        np.fill_diagonal(pdist, np.inf)
        nearest.append(pdist.argmin(axis=1))
    return np.hstack(nearest)

nearest_pse = find_nearest(reprs_pse, metric='cosine')


# BASELINES
codes_onehot = np.zeros((len(dataset), dataset.num_codes()), dtype=bool)
for i, (codes, _) in enumerate(dataset):
    codes_onehot[i, codes] = 1
    
# PCA: pca of one-hode encoded codes
pca = PCA(n_components=out_dim) #  cohorts
reprs_pca = pca.fit_transform(codes_onehot) # (N, 9)
nearest_pca = find_nearest(reprs_pca, metric='euclidean')

# Hamming distance:
reprs_hamming = codes_onehot
nearest_hamming = find_nearest(reprs_hamming, metric='hamming')



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



# Disease Cohort Classification

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

labels = dataset.labels

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)



methods = {
    "PSE": reprs_pse,
    "PCA": reprs_pca,
}
method_acc = {}
method_auc = {}

for name, x in methods.item():

    kf = KFold(n_splits=10)
    score = []
    auc = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = MLPClassifier().fit(x_train, y_train)
        # y_hat  = clf.predict(x_test)
        y_hat_prob = clf.predict_proba(x_test)
        auc.append(roc_auc_score(y_test, y_hat_prob))
        score.append(clf.score(x_test, y_test)) # provides mean accuracy

    method_acc[name] = score
    method_auc[name] = auc




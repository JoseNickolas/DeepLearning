from pathlib import Path
import json

import numpy as np
from sklearn.metrics import pairwise_distances_chunked

@np.vectorize
def has_intersection(x, y):
    """weather 1d arrays x and y has any common elements"""
    return len(set(x) & set(y)) > 0
    
    
def pairwise_intersect(x):
    """weather pairs of arrays in x has intersection or not

    Args:
        x (list[list]): list of lists with arbitrary length. len(x) is N.
    Returns:
        ndarray of shape (N, N). Element [i, j] is 1 if x[i] and x[j] have
        a common elemnt.
    """
    x = np.array(x, dtype=object)
    return has_intersection(x, x[:, None])
    
    
def save_config(config, path):
    path = Path(path) / 'config.json'
    with open(path, 'w') as file:
        file.write(json.dumps(config))
    
    
def find_nearest(representation, metric):
    nearest = []
    for pdist in pairwise_distances_chunked(representation, metric=metric, n_jobs=-1):
        if metric == 'cosine':
            pdist = 1 - pdist
        np.fill_diagonal(pdist, np.inf)
        nearest.append(pdist.argmin(axis=1))
    return np.hstack(nearest)
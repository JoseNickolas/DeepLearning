import numpy as np

def _has_intersection(x, y):
    """weather 1d arrays x and y has any common elements"""
    return len(set(x) & set(y)) > 0
    
has_intersection = np.vectorize(_has_intersection)

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
    
    
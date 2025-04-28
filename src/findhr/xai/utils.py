import numpy as np


def rank_list(vector):
    """
    returns ndarray containing rank(i) for documents at position i
    """
    temp = vector.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(vector) + 1)

    return ranks
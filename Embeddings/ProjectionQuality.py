import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.metrics import pairwise_distances

def projection_quality( X1, X2 ):
    # pairwise distances
    D1 = csr_matrix(pairwise_distances(X1))
    D2 = csr_matrix(pairwise_distances(X2))
    
    # minimum spanning tree
    MST1 = minimum_spanning_tree(D1)
    MST2 = minimum_spanning_tree(D2)
    
    max_sd = 0.5

    # global distance distribution factor
    Q_dd_1 = D1.toarray().std() / max_sd
    Q_dd_2 = D2.toarray().std() / max_sd
    
    # average graph edge weights
    w_A_1 = MST1[MST1.nonzero()].mean()
    w_A_2 = MST2[MST2.nonzero()].mean()
    
    # limit threshold
    D1_full = D1.A
    np.fill_diagonal(D1_full, np.infty)
    limit1  = np.maximum(D1_full.min(0), np.full(D1_full.shape[0], w_A_1)) + w_A_1 * Q_dd_1
    D1_limit   = np.tile(limit1.reshape(-1,1),  (1,D1.shape[0]))
    
    D2_full = D2.A
    np.fill_diagonal(D2_full, np.infty)
    limit2  = np.maximum(D2_full.min(0), np.full(D2_full.shape[0], w_A_2)) + w_A_2 * Q_dd_2
    D2_limit = np.tile(limit2.reshape(-1,1),(1,D2.shape[0]))
    
    # EMST
    EMST1_list = np.maximum(MST1.A, (D1 < D1_limit).astype(int)).nonzero()
    EMST2_list = np.maximum(MST2.A, (D2 < D2_limit).astype(int)).nonzero()
    
    N = D1.shape[0]
    EMST1 = csr_matrix(([1]*len(EMST1_list[0]), EMST1_list), shape=(N, N))
    EMST2 = csr_matrix(([1]*len(EMST2_list[0]), EMST2_list), shape=(N, N))
    
    def precision(i):
        return (EMST1[i].A.astype(bool) & EMST2[i].A.astype(bool)).sum() / EMST2[i].sum()
    
    def recall(i):
        return (EMST1[i].A.astype(bool) & EMST2[i].A.astype(bool)).sum() / EMST1[i].sum()
    
    def f1(i):
        prec = precision(i)
        rec = recall(i)
        return prec * rec / (prec + rec) * 2
    
    return [f1(i) for i in range(X1.shape[0])]

    

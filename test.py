import pandas as pd
from pandas import read_csv
import numpy as np
import scipy as sp
from algorithms import *

male_edges = {(int(str.removeprefix(r[1], "m")), int(str.removeprefix(r[2], "m"))): int(r[3]) 
                for r in read_csv("male_connectome_graph.csv").itertuples()}
female_edges = {(int(str.removeprefix(r[1], "f")), int(str.removeprefix(r[2], "f"))): int(r[3])
                for r in read_csv("female_connectome_graph.csv").itertuples()}
matching = {int(str.removeprefix(r[1], "m")): int(str.removeprefix(r[2], "f")) for r in read_csv("benchmark.csv").itertuples()}

def get_csr_matrix(edges):

    data = []
    coords = ([],[])

    for (u,v), weight in edges.items():
        coords[0].append(u-1)
        coords[1].append(v-1)
        data.append(weight)

    M=sp.sparse.coo_array((data,coords)).astype(dtype=float).tocsr()

    return M

def score(M, F, mapping):

    if isinstance(M,sp.sparse.csr_array):
        S=sp.sparse.csr_array.minimum(M,F[mapping[:,None],mapping])
        return S.sum()
    else:
        return np.minimum(M,F[mapping[:,None],mapping]).sum()
    
def get_best_connected(X, m:int):
    total_degrees = (X!=0).sum(axis=1)+(X!=0).sum(axis = 0)
    index_order = np.argsort(total_degrees)
    U = index_order[-m:]
    return X[U[:,None],U[None,:]], index_order[-m:]

M = get_csr_matrix(male_edges)
F = get_csr_matrix(female_edges)
n = M.shape[0]

benchmark_mapping = np.zeros(n,dtype=int)
for u,v in matching.items():
    benchmark_mapping[u-1] = v-1

A = M.todense()
B = F.todense()

m=100
H_M,M_positions = get_best_connected(M,m)
H_F,F_positions = get_best_connected(F,m)
H_M = H_M.todense()
H_F = H_F.todense()
np.count_nonzero(H_M)/m**2, np.count_nonzero(H_F)/m**2

from algorithms import random_swaps
H_mapping = np.arange(m)

print(f"Starting score={score(H_M,H_F,H_mapping)}")
random_swaps(H_M, H_F,H_mapping,steps=100000)
print(f"Final score={score(H_M,H_F,H_mapping)}")

from algorithms import greedy_mapping
start_mapping=-np.ones(n,dtype=int)
start_mapping[M_positions] = F_positions[H_mapping]
mapping = greedy_mapping(A,B,start_mapping)
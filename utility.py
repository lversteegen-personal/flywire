from pandas import read_csv
import scipy as sp
import numpy as np

male_edges = {(int(str.removeprefix(r[1], "m")), int(str.removeprefix(r[2], "m"))): int(r[3]) 
                for r in read_csv("male_connectome_graph.csv").itertuples()}
female_edges = {(int(str.removeprefix(r[1], "f")), int(str.removeprefix(r[2], "f"))): int(r[3])
                for r in read_csv("female_connectome_graph.csv").itertuples()}
matching = {int(str.removeprefix(r[1], "m")): int(str.removeprefix(r[2], "f")) for r in read_csv("benchmark.csv").itertuples()}
alignment = 0

#for male_nodes, edge_weight in male_edges.items():
#  female_nodes = (matching[male_nodes[0]], matching[male_nodes[1]])
#  alignment += min(edge_weight, female_edges.get(female_nodes, 0))

#print(f"{alignment=}")

def get_csr_matrix(edges):

    data = []
    coords = ([],[])

    for (u,v), weight in edges.items():
        coords[0].append(u-1)
        coords[1].append(v-1)
        data.append(weight)

    M=sp.sparse.coo_array((data,coords)).astype(dtype=int).tocsr()

    return M

def random_graph(n, max=1, rng = None):

    if rng is None:
        rng = np.random.default_rng()
    G = rng.integers(max+1,size=(n,n))
    G[np.arange(n),np.arange(n)] = 0
    return G

def shuffle_graph(G,rng=None):

    if rng is None:
        rng = np.random.default_rng()
    n = G.shape[0]
    shuffle = rng.permutation(n)
    H = G[shuffle[:,None],shuffle[None,:]]
    return H, invert(shuffle)

def get_best_connected(X, m:int):
    total_degrees = (X!=0).sum(axis=1)+(X!=0).sum(axis = 0)
    index_order = np.argsort(total_degrees)
    U = index_order[-1:-m-1:-1]
    return X[U[:,None],U[None,:]], U

def get_highest_out_degrees(X, m:int):
    degrees = (X!=0).sum(axis=1)
    index_order = np.argsort(degrees)
    U = index_order[-1:-m-1:-1]
    return X[U[:,None],U[None,:]], U

def invert(mapping):

    inverse = np.zeros(mapping.size,dtype=int)
    inverse[mapping] = np.arange(mapping.size)
    return inverse

def score(G, H, mapping, require_surjective = True):

    if require_surjective:
        check = np.zeros(H.shape[0],dtype=bool)
        check[mapping] = True
        if check.min() == False:
            raise Exception("Mapping to be scored is not surjective!")

    if isinstance(G,sp.sparse.csr_array):
        S=sp.sparse.csr_array.minimum(G,H[mapping[:,None],mapping])
        return S.sum()
    else:
        return np.minimum(G,H[mapping[:,None],mapping[None,:]]).sum()

M = get_csr_matrix(male_edges)
F = get_csr_matrix(female_edges)

n=M.shape[0]

benchmark_mapping = np.zeros(n,dtype=int)
for u,v in matching.items():
    benchmark_mapping[u-1] = v-1


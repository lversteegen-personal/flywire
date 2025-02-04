import numpy as np
import scipy as sp

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

def random_graph_dist(n, p=1,sample_source=None,rng=None):

    if sample_source is None:
        sample_source = lambda:1
    if rng is None:
        rng = np.random.default_rng()
    m = rng.binomial(n**2,p)
    edges = rng.choice(n**2,size=m,replace=False)
    G = np.zeros(n**2,dtype=int)
    
    for i in range(m):
        G[edges[i]] = sample_source()
    
    G = G.reshape((n,n))    

    G[np.arange(n),np.arange(n)] = 0
    return G

def get_cycles(permutation):

    cycles = []
    visited = set()

    for i in range(len(permutation)):
        if i in visited:
            continue

        c = []
        j=i
        while True:
            visited.add(j)
            c.append(j)
            if permutation[j] == i:
                break
            else:
                j = permutation[j]

        cycles.append(np.array(c))

    return cycles

def shuffle_cycles(permutation,rng=None):

    if rng is None:
        rng = np.random.default_rng()

    cycles = get_cycles(permutation)
    new_permutation = permutation.copy()

    for c in cycles:
        new_permutation[c] = rng.permutation(c)

    return new_permutation

def random_extension(n,partial_mapping_from, partial_mapping_to,rng = None):

    if rng is None:
        rng = np.random.default_rng()

    extension = np.arange(n)

    permute_source = np.delete(extension,partial_mapping_from)
    permute_target = np.delete(extension,partial_mapping_to)
    extension[permute_source] = rng.permutation(permute_target)
    extension[partial_mapping_from] = partial_mapping_to

    return extension

def shuffle_map(map, stable=None,rng = None):

    if rng is None:
        rng = np.random.default_rng()
    n = len(map)

    shuffle = map.copy()
    if stable is not None:
        permute_source = np.delete(np.arange(n),stable)
        permute_target = np.delete(np.arange(n),map[stable])
        shuffle[permute_source] = rng.permutation(permute_target)
    else:
        shuffle = rng.permutation(n)

    return shuffle

def shuffle_graph(G,stable=None,rng=None):

    shuffle = shuffle_map(np.arange(G.shape[0]),stable=stable,rng=rng)

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

#Don't forget to invert!
def save_as_csv(mapping, file_name):
    """Remember to invert the mapping if necessary!"""

    rows = [("Male Node ID","Female Node ID")]
    rows.extend(zip(["m"+str(i+1) for i in range(len(mapping))],["f"+str(j+1) for j in mapping]))
    np.savetxt(file_name,rows,fmt="%s",delimiter=",")
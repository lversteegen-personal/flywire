import numpy as np

rng = np.random.default_rng()

def swap_score(M,F, mapping, u, v):

    f_u = mapping[u]
    f_v = mapping[v]

    u_row = M[u,:]
    u_col = M[:,u]
    v_row = M[v,:]
    v_col = M[:,v]
    f_u_row = F[[f_u],mapping]
    f_u_col = F[mapping,[f_u]]
    f_v_row = F[[f_v],mapping]
    f_v_col = F[mapping,[f_v]]

    old_score = np.minimum(u_row,f_u_row).sum() + np.minimum(u_col,f_u_col).sum() + np.minimum(v_row,f_v_row).sum() + np.minimum(v_col,f_v_col).sum()
    old_score = old_score - min(M[u,v],F[f_u,f_v]) - min(M[v,u],F[f_v,f_u]) - min(M[u,u],F[f_u,f_u]) - min(M[v,v],F[f_v,f_v])
    new_score = np.minimum(u_row,f_v_row).sum() + np.minimum(u_col,f_v_col).sum() + np.minimum(v_row,f_u_row).sum() + np.minimum(v_col,f_u_col).sum()
    new_score = new_score + min(M[u,v],F[f_v,f_u])+min(M[v,u],F[f_u,f_v]) - min(M[u,u],F[f_v,f_u]) - min(M[u,u],F[f_u,f_v]) - \
        min(M[v,v],F[f_v, f_u]) - min(M[v,v],F[f_u,f_v])

    return new_score - old_score

def random_swaps(A, B, mapping, steps,verbose=True):

    n = A.shape[0]

    u = rng.integers(n,size=steps)
    v = (rng.integers(n-1,size=steps) + u + 1)%n

    for i in range(steps):
        if swap_score(A,B,mapping, u[i],v[i]) > 0:
            tmp = mapping[u[i]]
            mapping[u[i]] = mapping[v[i]]
            mapping[v[i]] = tmp

        if verbose:
            print(f"{i=}",end="\r")

def greedy_mapping(A, B, start_mapping = None, similarity = None, c=0.1):
    n = A.shape[0]

    if start_mapping is None:
        mapping = -np.ones(n,dtype=int)
        unmapped_source = np.ones(n,dtype=bool)
        unmapped_target = np.ones(n,dtype=bool)
    else:
        mapping = start_mapping.copy()
        unmapped_source = mapping < 0
        unmapped_target = np.ones(n,dtype=bool)
        unmapped_target[mapping[np.logical_not(unmapped_source)]] = False

    priority = c*np.log1p(A.sum(axis=1))
    if similarity is None:
        if start_mapping is None:
            similarity = np.minimum(A.sum(axis=1).reshape(-1,1),B.sum(axis=1).reshape(1,-1))
            similarity += np.minimum(A.sum(axis=0).reshape(-1,1),B.sum(axis=0).reshape(1,-1))
            np.log1p(similarity,out=similarity)
        else:
            similarity = np.zeros((n,n),dtype=np.int64)
            for u in np.flatnonzero(np.logical_not(unmapped_source)):
                v = mapping[u]

                u_out = np.flatnonzero(A[u,:])
                u_out = u_out[unmapped_source[u_out]]
                v_out = np.flatnonzero(B[v,:])  
                v_out = v_out[unmapped_target[v_out]]

                min_out = np.minimum.outer(A[[u],u_out],B[[v],v_out])
                np.add.at(similarity,(u_out[:,None],v_out[None,:]),min_out)

                u_in = np.flatnonzero(A[:,u])
                u_in = u_in[unmapped_source[u_in]]
                v_in = np.flatnonzero(B[:,v])
                v_in = v_in[unmapped_target[v_in]]

                min_in = np.minimum.outer(A[u_in,[u]],B[v_in,[v]])
                np.add.at(similarity,(u_in[:,None],v_in[None,:]),min_in)

            priority[np.logical_not(unmapped_source)] = -1
            similarity[:,np.logical_not(unmapped_target)] = -1


    for i in range(np.count_nonzero(unmapped_source)):

        u = priority.argmax()
        v = similarity[u,:].argmax()
        
        u_out = np.flatnonzero(A[u,:])
        u_out = u_out[unmapped_source[u_out]]
        v_out = np.flatnonzero(B[v,:])  
        v_out = v_out[unmapped_target[v_out]]

        min_out = np.minimum.outer(A[[u],u_out],B[[v],v_out])
        np.add.at(similarity,(u_out[:,None],v_out[None,:]),min_out)

        u_in = np.flatnonzero(A[:,u])
        u_in = u_in[unmapped_source[u_in]]
        v_in = np.flatnonzero(B[:,v])
        v_in = v_in[unmapped_target[v_in]]

        min_in = np.minimum.outer(A[u_in,[u]],B[v_in,[v]])
        np.add.at(similarity,(u_in[:,None],v_in[None,:]),min_in)

        unmapped_source[u] = False
        unmapped_target[v] = False
        mapping[u]=v
        priority[u_out] += A[[u],u_out]
        priority[u_in] += A[u_in,[u]]
        similarity[:,v] = -1
        similarity[u,:] = -1
        priority[u] = -1

        print(f"{i=}",end="\r")

    return mapping
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

def greedy_max(A, B, start_mapping = None, similarity = None, c=0.1, verbose = True, injective = True):
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
            similarity = np.log1p(similarity)
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

            priority = similarity.max(axis=1)
            priority[np.logical_not(unmapped_source)] = -1
            if injective:
                similarity[:,np.logical_not(unmapped_target)] = -1


    for i in range(np.count_nonzero(unmapped_source)):

        u = priority.argmax()
        v = similarity[u,:].argmax()
        
        u_out = np.flatnonzero(A[u,:])
        v_out = np.flatnonzero(B[v,:])  
        if injective:
            u_out = u_out[unmapped_source[u_out]]
            v_out = v_out[unmapped_target[v_out]]

        min_out = np.minimum.outer(A[[u],u_out],B[[v],v_out])
        np.add.at(similarity,(u_out[:,None],v_out[None,:]),min_out)

        u_in = np.flatnonzero(A[:,u])
        v_in = np.flatnonzero(B[:,v])

        if injective:
            u_in = u_in[unmapped_source[u_in]]
            v_in = v_in[unmapped_target[v_in]]

        min_in = np.minimum.outer(A[u_in,[u]],B[v_in,[v]])
        np.add.at(similarity,(u_in[:,None],v_in[None,:]),min_in)

        unmapped_source[u] = False
        unmapped_target[v] = False
        mapping[u]=v

        u_out = u_out[unmapped_source[u_out]]
        v_out = v_out[unmapped_target[v_out]]
        u_in = u_in[unmapped_source[u_in]]
        v_in = v_in[unmapped_target[v_in]]
        if (len(u_out)>0 and len(v_out)>0):
            priority[u_out] = np.maximum(priority[u_out],similarity[u_out[:,None],v_out[None,:]].argmax(axis=1))
        if (len(u_in)>0 and len(v_in)>0):
            priority[u_in] = np.maximum(priority[u_in],similarity[u_in[:,None],v_in[None,:]].argmax(axis=1))
        if injective:
            similarity[:,v] = -1
            similarity[u,:] = -1
            
        priority[u] = -1

        if verbose:
            print(f"{i=}",end="\r")

    return mapping

def find_best_move(G,H, mapping, u, mask = None):

    m = G.shape[0]
    u_row = G[u,:]
    u_rnz = np.flatnonzero(u_row)
    u_row = u_row[u_rnz]
    u_col = G[:,u]
    u_cnz = np.flatnonzero(u_col)
    u_col = u_col[u_cnz]

    if mask is None:
        n = H.shape[0]
        X = H[:,mapping[None,u_rnz]].reshape((n,len(u_rnz)))
        Y = H[mapping[u_cnz,None],:].reshape((len(u_cnz),n))
    else:
        X = H[mask,mapping[None,u_rnz]]
        n = X.shape[0]
        X = X.reshape((n,len(u_rnz)))
        Y = H[mapping[u_cnz,None],mask].reshape((len(u_cnz),n))

    scores = np.minimum(u_row[None,:],X).sum(axis=1) + np.minimum(u_col[:,None],Y).sum(axis=0)
    scores -= np.minimum(G[u,u],H.diagonal())

    return scores

#The mask says which x in [n] are valid choices for mapping[u]. G and H must have 0-diagonal
def make_best_swap(G,H, mapping, u, scores_by_vertex, mask = None):

    if mask is None:
        m = G.shape[0]
        n = H.shape[0]
    else:
        in_image = np.zeros(len(mask),dtype=bool)
        in_image[mapping] = True
        H_mask = np.logical_and(in_image,mask)
        G_mask = mask[mapping]
        m = np.count_nonzero(G_mask)
        n = np.count_nonzero(H_mask)

    row = G[u,:]
    rnz = np.flatnonzero(row)
    row = row[rnz]
    col = G[:,u]
    cnz = np.flatnonzero(col)
    col = col[cnz]

    if mask is None:
        X = H[mapping[:,None],mapping[None,rnz]].reshape((n,len(rnz)))
        Y = H[mapping[cnz,None],mapping[None,:]].reshape((len(cnz),n))
    else:
        X = H[H_mask,mapping[None,rnz]].reshape((n,len(rnz)))
        Y = H[mapping[cnz,None],H_mask].reshape((len(cnz),n))

    u_scores = np.minimum(row[None,:],X).sum(axis=1) + np.minimum(col[:,None],Y).sum(axis=0)

    f_u = mapping[u]

    row = H[f_u,mapping]
    rnz = np.flatnonzero(row)
    row = row[rnz]
    col = H[mapping,f_u]
    cnz = np.flatnonzero(col)
    col = col[cnz]

    if mask is None:
        X = G[:,rnz[None,:]].reshape((m,len(rnz)))
        Y = G[cnz[:,None],:].reshape((len(cnz),m))
    else:
        X = G[G_mask,rnz[None,:]].reshape((m,len(rnz)))
        Y = G[mapping[cnz,None],G_mask].reshape((len(cnz),m))

    f_u_scores = np.minimum(row[None,:],X).sum(axis=1) + np.minimum(col[:,None],Y).sum(axis=0)

    if mask is None:
        scores = u_scores + f_u_scores + np.minimum(G[u,:],H[mapping,f_u])+np.minimum(G[:,u],H[f_u,mapping])
        scores -= scores_by_vertex
        scores -= scores_by_vertex[u]
        scores += np.minimum(G[u,:],H[f_u,mapping])+np.minimum(G[:,u],H[mapping,f_u])
        #The actual change will be 2*scores[w], but this does not change which vertex we will choose for the swap.
    else:
        scores = u_scores + f_u_scores + np.minimum(G[u,G_mask],H[mapping[G_mask],f_u])+np.minimum(G[G_mask,u],H[f_u,mapping[G_mask]])
        scores -= scores_by_vertex[G_mask]
        scores -= scores_by_vertex[u]
        scores += np.minimum(G[u,G_mask],H[f_u,mapping[G_mask]])+np.minimum(G[G_mask,u],H[mapping[G_mask],f_u])

    w = scores.argmax()
    if scores[w] > 0:

        f_w = mapping[w]

        scores_by_vertex += np.minimum(G[:,u],H[mapping,f_w])+np.minimum(G[u,:],H[f_w,mapping])+np.minimum(G[:,w],H[mapping,f_u])+np.minimum(G[w,:],H[f_u,mapping])
        scores_by_vertex -= np.minimum(G[:,u],H[mapping,f_u])+np.minimum(G[u,:],H[f_u,mapping])+np.minimum(G[:,w],H[mapping,f_w])+np.minimum(G[w,:],H[f_w,mapping])

        scores_by_vertex[u] = u_scores[w]+ np.minimum(G[u,w],H[f_w,f_u])+np.minimum(G[w,u],H[f_u,f_w])
        scores_by_vertex[w] = f_u_scores[w]+ np.minimum(G[u,w],H[f_w,f_u])+np.minimum(G[w,u],H[f_u,f_w])

        mapping[u] = f_w
        mapping[w] = f_u

        return scores[w]
    
    else:
        return 0
    
def get_scores_by_vertex(G,H,mapping):

    m = G.shape[0]
    scores = np.minimum(G,H[mapping[:,None],mapping[None,:]])
    return scores.sum(axis=0)+scores.sum(axis=1)
    
def move_or_swap(G,H, mapping, u, scores_by_vertex, usage, penalty_gradient, rng =None):

    m = G.shape[0]
    n = H.shape[0]
    f_u = mapping[u]

    row = G[u,:]
    rnz = np.flatnonzero(row)
    row = row[rnz]
    col = G[:,u]
    cnz = np.flatnonzero(col)
    col = col[cnz]

    X = H[:,mapping[None,rnz]].reshape((n,len(rnz)))
    Y = H[mapping[cnz,None],:].reshape((len(cnz),n))

    base_scores = np.minimum(row[None,:],X).sum(axis=1) + np.minimum(col[:,None],Y).sum(axis=0)
    u_scores = base_scores[mapping]

    move_scores = base_scores - base_scores[f_u]
    move_scores -= penalty_gradient(usage) 
    move_scores += penalty_gradient(usage[f_u]-1)
    move_scores[f_u] = 0

    row = H[f_u,mapping]
    rnz = np.flatnonzero(row)
    row = row[rnz]
    col = H[mapping,f_u]
    cnz = np.flatnonzero(col)
    col = col[cnz]

    X = G[:,rnz[None,:]].reshape((m,len(rnz)))
    Y = G[cnz[:,None],:].reshape((len(cnz),m))

    f_u_scores = np.minimum(row[None,:],X).sum(axis=1) + np.minimum(col[:,None],Y).sum(axis=0)

    swap_scores = u_scores + f_u_scores + np.minimum(G[u,:],H[mapping,f_u])+np.minimum(G[:,u],H[f_u,mapping])
    swap_scores -= scores_by_vertex
    swap_scores -= scores_by_vertex[u]
    swap_scores += np.minimum(G[u,:],H[f_u,mapping])+np.minimum(G[:,u],H[mapping,f_u])

    if rng is None:
        x = move_scores.argmax()
        w = swap_scores.argmax()
    else:
        max_value = move_scores.max()
        max_indices=np.flatnonzero(move_scores==max_value)
        x = rng.choice(max_indices)
        max_value = swap_scores.max()
        max_indices=np.flatnonzero(swap_scores==max_value)
        w = rng.choice(max_indices)
        
    if swap_scores[w] > 0 and swap_scores[w] >= move_scores[x]:

        f_w = mapping[w]

        scores_by_vertex += np.minimum(G[:,u],H[mapping,f_w])+np.minimum(G[u,:],H[f_w,mapping])+np.minimum(G[:,w],H[mapping,f_u])+np.minimum(G[w,:],H[f_u,mapping])
        scores_by_vertex -= np.minimum(G[:,u],H[mapping,f_u])+np.minimum(G[u,:],H[f_u,mapping])+np.minimum(G[:,w],H[mapping,f_w])+np.minimum(G[w,:],H[f_w,mapping])

        scores_by_vertex[u] = u_scores[w]+ np.minimum(G[u,w],H[f_w,f_u])+np.minimum(G[w,u],H[f_u,f_w])
        scores_by_vertex[w] = f_u_scores[w]+ np.minimum(G[u,w],H[f_w,f_u])+np.minimum(G[w,u],H[f_u,f_w])

        mapping[u] = f_w
        mapping[w] = f_u

        return swap_scores[w],True
    elif move_scores[x] > 0:

        scores_by_vertex += np.minimum(G[:,u],H[mapping,x])+np.minimum(G[u,:],H[x,mapping]) - np.minimum(G[:,u],H[mapping,f_u])- np.minimum(G[u,:],H[f_u,mapping])
        
        scores_by_vertex[u] = base_scores[x]

        mapping[u] = x
        usage[f_u] -= 1
        usage[x] += 1

        return base_scores[x] - base_scores[f_u],True
    else:
        return 0,False

def move_swap_optimization(G,H,start_mapping=None,penalty_gradient=lambda x:0,rng=None,max_epochs = 50,verbose=False,randomize_argmax=False):

    N = G.shape[0]
    k = H.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    if start_mapping is None:
        mapping = rng.choice(k,size=N)
    else:
        mapping = np.copy(start_mapping)

    usage = np.zeros(k,dtype=np.int32)
    np.add.at(usage,mapping,1)
    scores_by_vertex = get_scores_by_vertex(G,H,mapping)
    score_track=scores_by_vertex.sum()//2

    for i in range(max_epochs):

        outer_improvement = False
        for j,u in enumerate(rng.permutation(N)):
            
            if randomize_argmax:
                change,improvement = move_or_swap(G,H,mapping,u,scores_by_vertex,usage,penalty_gradient=penalty_gradient,rng=rng)
            else:
                change,improvement = move_or_swap(G,H,mapping,u,scores_by_vertex,usage,penalty_gradient=penalty_gradient)
            score_track += change
            #if score_track != score(G,H,mapping,require_surjective=False) or scores_by_vertex.sum()//2 != score_track:
            #    raise Exception("Error!")
            outer_improvement = improvement or outer_improvement
            
            if verbose and j%20==0:
                print(f"{i=}, {j=}, {score_track=}, unique={len(np.unique(mapping))}"+10*" ",end="\r")

        if not outer_improvement:
            break

    return mapping, score_track
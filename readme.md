# Flywire

This repository contains the experiments and algorithms I wrote for my submission for the FlyWire VNC Matching Challenge. A full description of the challenge may be found [here](https://codex.flywire.ai/app/vnc_matching_challenge), but the gist is that we are trying to quantifiy the similarity of two weighted directed graphs M=(V_M,E_M) and F=(V_F,E_F), representing neural connections in corresponding parts of the brain of a male and female fruitfly, respectively.

More precisely, we are challenged to find an (invertible!) map f: V_M->V_F such that score(f), which is defined as the sum of min(E_M(u,v),E_F(f(u),f(v))) over all ordered pairs (u,v) of vertices in V_M, is as large as possible. Due to the large number of vertices (18524), any algorithm we use to construct f must be highly efficient in both space and time.

## Connectome 

The data for the connections in M and F is not part of this repository, but it can be found on the [challenge homepage](https://codex.flywire.ai/app/vnc_matching_challenge). 

## Benchmark mapping and upper limits

The challenge team also provided a benchmark map f_0 (courtesy of Alexander Bates), that achieves a score of roughly 5.15 million, and I worked under the assumption that any map f such that score(f) is substantially higher than score(f_0) warrants a submission for the challenge. The highest score that is theoretically possible is 8605294. This is because the sum of all weights in the female connectome is 8900885 and 295598 of that weight is tied up in loops. Curiously, in the male connectome loops only account for a total weight of 7, meaning that the score can be at most 8900885-295598+7=8605294.

# The Files

data_loader.py - Loads the male and female connectome as scipy sparse matrices M and F respectively. It also loads benchmark_mapping in form of a 1D numpy array.

utility.py - Contains functions that I found useful during experimentation, in particular for generating random graphs for testing new algorithms.

algorithms.py - Here I consolidated algorithms that I wanted to be able to access from different notebooks.

matching.ipynb - A notebook showing various experiments I made in search of a better mapping.

analysis.ipnyb - A notebook analysing some of the properties of the two graphs and the mapping between them.

cupy.ipynb - Since my algorithms were heavily vectorized, I could accelerate them by using a GPU. I think it is more standard to write code that is agnostic to whether it will be using cupy or numpy. However, there do seem to be certain differences in the API of the two frameworks (for example, the rng of cupy offers fewer methods), and because I had to rely on the free GPU minutes of Google Colab (and since iterating versions in my Github-Colab workflow is a pain), I could not afford to spend much time trying to write framework agnostic code. Instead, I set up a separate cupy notebook to use with A GPU in Colab. The functions in its cells are essentially copied from algorithms.py.

# Approaches and remarks

## Random swaps

The most naive approach is to swap the images f(u) and f(v) for randomly chosen vertices u and v, calculate the score after the swap and undo the swap if the new score is lower than the old one. 

## Greedy swaps

It is not necessary to recalculate the entire score for each swap, which is quite useful because the score calculation O(|E_M|+|E_F|). In fact, one can find the optimal swapping partner for a given vertex u in O(n*(d(u)+d(v))). On average this is O(|E_M|+|E_F|), which essentially means that we can try n swaps for the price of 1. What is more, calculating the optimal swap can be done in a numpy/cupy-friendly vectorized manner. The algorithm for finding an optimal swapping partner is implemented in make_best_swap.

## Why I think you cannot do much better than pair swaps

To figure out whether the mapping could be improved further, I tried to see whether the greedy-swap algorithm can learn the identity from one of the brains to themselves. I.e., we start with a random bijection V_F -> V_F and see how close we can get to the score of the identity by doing greedy swaps. I tried this a few times and it turns out that it learns the identity quite reliably. If it does not get stuck in a local maximum for this, it seems likely that there are few large-scale symmetries in either graph, so that the algorithm should not get stuck at a strong local optimum.

The competition is now over (Congratulations to the victorious Team "Old School"!) and while quite a few teams were able to get a slightly higher score, my best mapping is within 1% of the score of the winning team.

## Random graphs

Consider the general algorithmic of optimizing score_{G,H}(f)=sum_{u,v} min(E_G(u,v),E_H(f(u),f(v))) for two (secretly similar) graphs G and H on the same number of vertices. There are three ways in which the FlyWire dataset is a poor benchmark for such an algorithm.

1. It is just one particular pair of graphs with many approximate symmetries. An algorithm that works well for this pair might perform poorly for others.
2. In all likelihood, it is not possible to find or even certify the best possible mapping. That means, it is difficult to have confidence that a mapping is actually close to best possible.
3. The Flywire graphs are quite big so that experiments take a long time, especially with limited compute.

For these reasons I thought it was a good idea to try out my more sophisticated ideas on isomorphic random graphs. I.e., we sample a graph G according to some distribution, and then we permute its vertices randomly to obtain H. In this setup, finding the map f:V(G)->V(H) that maximizes score_{G,H}(f) reduces to the well-known [graph isomorphism problem](https://en.wikipedia.org/wiki/Graph_isomorphism_problem), which can be solved in quasi-polynomial time (in fact, with the exception of very pathological pairs (G,H) in linear time). However, we are of course still allowed to optimize score_{G,H} by methods that are valid for non-isomorphic graphs.

The greedy-swap algorithm seems to get stuck in local maxima for Erdős–Rényi–Gilbert random graphs, much more than for the connectome graphs. Greedy-swap tends to get stuck less if the edges of an ERG random graphs are given weights according to some slowly decaying distribution. This seems to confirm the heuristic that score_{G,H} is easier to optimize if G and H have more structure.

## Non-injective mappings

One approach I took to avoiding local maxima was to allow non-bijective maps f:V(G)->V(H) and then gradually smooth the mapping by increasing a penalty term that punishes many vertices in G being mapped to the same vertex in H (implemented in move_or_swap). Unfortunately, this did not lead to an improvement over the vanilla greedy mapping approach.

## Taking hints

When we actually no what the "correct" mapping f is, we can give "hints" to algorithms such as greedy swap by setting the initial mapping to use in the algorithm to partially agree with f. I believe that for an ERG random graph it is enough to give a hint for sqrt(n)/log n vertices for greedy_swap to find the optimal mapping with high probability and the practice roughly match this.

## Generating hints with committees / ensembles

I hoped that it would be possible to generate such hints without prior knowledge of the optimal mapping by a bootstrapping process. We let the algorithm run with different seeds for the same pair of graphs to obtain an "ensemble" of mappings f_1,...,f_k and then look whether there are any pairs of vertices (u,v) for which f_i(u)=v for significantly more f_i than expected. Initializing the algorithms with these pairs as a hint should also yield the optimal mapping as long as there are enough of them, but unfortunately, for ERG random graphs, the distribution of f_i(u) was extremly uniform for all u. For otherwise randomly drawn graphs this approach seemed to improve the success rate a little bit.
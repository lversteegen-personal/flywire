#Flywire

This repository contains the experiments and algorithms I wrote for my submission for the FlyWire VNC Matching Challenge. A full description of the challenge may be found [here](https://codex.flywire.ai/app/vnc_matching_challenge), but the gist is that we are trying to quantifiy the similarity of two weighted directed graphs M=(V_M,E_M) and F=(V_F,E_F), representing neural connections in corresponding parts of the brain of a male and female fruitfly, respectively.

More precisely, we are challenged to find an (invertible!) map f: V_M->V_F such that score(f), which is defined as the sum of min(E_M(u,v),E_F(f(u),f(v))) over all ordered pairs (u,v) of vertices in V_M, is as large as possible. Due to the large number of vertices (18524), any algorithm we use to construct f must be highly efficient in both space and time.

##Connectome 

The data for the connections in M and F is not part of this repository, but it can be found on the [challenge homepage](https://codex.flywire.ai/app/vnc_matching_challenge). 

#Benchmark mapping and upper limits

The challenge team also provided a benchmark map f_0 (courtesy of Alexander Bates), that achieves a score of roughly 5.15 million, and I worked under the assumption that any map f such that score(f) is substantially higher than score(f_0) warrants a submission for the challenge.

##Random swaps

##Greedy mapping

##Greedy swaps

##Why I think you cannot do much better (check this again!)
To figure out whether the mapping could be improved further, I tried to see whether the greedy-swap algorithm can learn the identity from one of the brains to themselves. I.e., we start with a random bijection V_F -> V_F and see how close we can get to the score of the identity by doing greedy swaps. I tried this a few times and it turns out that it learns the identity quite reliably. If it does not get stuck in a local maximum for this, it seems likely that there are few large-scale symmetries in either graph, so that the algorithm should not get stuck at a strong local optimum.
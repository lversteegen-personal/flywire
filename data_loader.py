from pandas import read_csv
import numpy as np
from utility import get_csr_matrix

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

M = get_csr_matrix(male_edges)
F = get_csr_matrix(female_edges)

n=M.shape[0]

benchmark_mapping = np.zeros(n,dtype=int)
for u,v in matching.items():
    benchmark_mapping[u-1] = v-1
import community
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import igraph as ig
import scipy.sparse as sp
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree




data = np.loadtxt("OurData/facebook_combined.txt")
file = open("OurData/facebook_group.txt")
print("data shape: ", data.shape)


G = ig.Graph()
G.add_vertices(4039)
edge_list = []
weight_list = []

for i in tqdm(range(data.shape[0])):
    n1 = int(data[i, 0])
    n2 = int(data[i, 1])
    # n1 and n2 are both in node_dict
    if (n1, n2) not in edge_list and (n2,n1) not in edge_list and n1!=n2:
        edge_list.append((n1, n2))
        weight_list.append(1.0)

G.add_edges(edge_list)
G.es['weight'] = weight_list


# input()



youtube_SN = sp.csr_matrix(G.get_adjacency().data)
youtube_SN = youtube_SN + youtube_SN.T
youtube_SN[youtube_SN > 1] = 1
span_N = youtube_SN.shape[0]
# lcc = largest_connected_components(youtube_SN)
# youtube_SN = youtube_SN[lcc,:][:,lcc]

val_share = 0.15
test_share = 0.00
seed = 481516234

# train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(
#     youtube_SN, val_share, test_share, seed, undirected=True, connected=True, asserts=True)

train_graph = youtube_SN

# 拿lcc後的Graph做partition
partition = community.best_partition(nx.from_numpy_matrix(train_graph.toarray()))
# partition = community.best_partition(nx.Graph(G.get_edgelist()))
print(max(partition.values()))
print(nx.info(nx.from_numpy_matrix(train_graph.toarray())))
print("======================================")




# input()




file = open("OurData/facebook_group.txt")
lines = file.readlines()
file.close()
node_list = []

for i in range(193): # use top K communitys nodes
    lines[i] = lines[i].replace("\n", "").split("\t")
    lines[i] = lines[i][1:]
    node_list.append(lines[i])
    
print(node_list)

ground_truth = np.zeros((train_graph.shape[0]))
for i in range(len(node_list)):
    for j in range(len(node_list[i])):
        ground_truth[int(node_list[i][j])] = i

        # ground_truth[node_dict[int(node_list[i][j])]] = i

label = ground_truth
keys, values = zip(*partition.items())
print(len(list(keys)))
print(len(list(values)))
pred = np.array(list(values))
print("ARI: ", adjusted_rand_score(label, pred))
print("NMI: ", normalized_mutual_info_score(label, pred))
print("F1 score micro: ", f1_score(label, pred, average='micro'))
print("F1 score macro: ", f1_score(label, pred, average='macro'))



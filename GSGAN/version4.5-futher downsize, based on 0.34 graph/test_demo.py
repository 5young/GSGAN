from netgan.netgan import *
import tensorflow as tf
from netgan import utils
import scipy.sparse as sp
import scipy
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
from tqdm import tqdm
import time
import igraph as ig
import argparse
import warnings
import os

print("==================================================================================")
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
warnings.filterwarnings(module='numpy*', action='ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', SparseEfficiencyWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', type=str, default='', help='experiment dir name')
opt = parser.parse_args()
K = 50
SMALLER_K = 45

# =====================================
# read facebook_combined social network
# =====================================
if  os.path.exists("matrix/1youtube_SN.npz"):
    print("*** loading dataset matrix ***")
    youtube_SN = scipy.sparse.load_npz('matrix/youtube_SN.npz')
    _N = youtube_SN.shape[0]

else:
    data = np.loadtxt("OurData/Youtube/com-youtube.ungraph.txt")
    file = open("OurData/Youtube/com-youtube.top5000.cmty.txt")

    num_of_top_K_comm = 100

    lines = file.readlines()
    file.close()
    node_list = []

    for i in range(num_of_top_K_comm): # use top K communitys nodes
        lines[i] = lines[i].replace("\n", "").split("\t")
        node_list.append(lines[i])

    node_list = [j for sub in node_list for j in sub]
    node_list = np.unique(np.array(node_list, dtype=int)).tolist() # set unique
    indices = [i for i in range(len(node_list))] # set node indices
    node_dict = dict(zip(node_list, indices)) # merge node_list and indices
    

    G = ig.Graph()
    G.add_vertices(len(indices))
    edge_list = []
    weight_list = []

    for i in tqdm(range(data.shape[0])):
        n1 = int(data[i, 0])
        n2 = int(data[i, 1])
        # n1 and n2 are both in node_dict
        if n1 in node_dict.keys() and n2 in node_dict.keys() and n1!=n2:
            if (node_dict[n1], node_dict[n2]) not in edge_list and (node_dict[n1], node_dict[n2]) not in edge_list:
                edge_list.append((node_dict[n1], node_dict[n2]))
                weight_list.append(1.0)

    G.add_edges(edge_list)
    G.es['weight'] = weight_list


    youtube_SN = sp.csr_matrix(G.get_adjacency().data)
    youtube_SN = youtube_SN + youtube_SN.T
    youtube_SN[youtube_SN > 1] = 1
    span_N = youtube_SN.shape[0]
    lcc = utils.largest_connected_components(youtube_SN)
    # youtube_SN = youtube_SN[lcc,:][:,lcc]
    _N = youtube_SN.shape[0]
    _E = youtube_SN.sum()/2

    print("N: ", _N, "// E: ", _E)

    # scipy.sparse.save_npz('matrix/youtube_SN.npz', youtube_SN)


val_share = 0.15
test_share = 0.00
seed = 481516234
train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(
	youtube_SN, val_share, test_share, seed, undirected=True, connected=True, asserts=True)


# train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
train_graph = youtube_SN
assert (train_graph.toarray() == train_graph.toarray().T).all()

# ======================================
#      Prepare the ground community
# ======================================
file = open("OurData/Youtube/com-youtube.top5000.cmty.txt")
lines = file.readlines()
file.close()
node_list = []

for i in range(num_of_top_K_comm): # use top K communitys nodes
    lines[i] = lines[i].replace("\n", "").split("\t")
    node_list.append(lines[i])
    
ground_truth_community = np.zeros((train_graph.shape[0]))
for i in range(len(node_list)):
    for j in range(len(node_list[i])):
        if node_dict[int(node_list[i][j])] < train_graph.shape[0]: # 沒搞懂這行是幹嘛用的, 不過好像都會符合這個標準, 忘了當初為何做這個檢查
            ground_truth_community[node_dict[int(node_list[i][j])]] = i



print()
print()
print()
print()
print()


# ===========================
# fb train graph processing
# ===========================
while True:
    gra = train_graph.toarray()
    _N = gra.shape[0]
    print("#############    Youtube social network    #############")
    print("*** train_graph shape: ", gra.shape)
    print("*** train graph edges: ", gra.sum())
    print("==================================================================================")

    rw_len = 40
    batch_size = 16
    fb_walker = utils.RandomWalker(train_graph, rw_len, p=10, q=1, batch_size=batch_size)
    rws = fb_walker.walk().__next__()
    a = utils.score_matrix_from_random_walks(rws, _N).tocsr().toarray()
    a[a>1] = 1
    print("*** sample rw as train_graph shape: ", a.shape)
    print("*** sample rw as train_graph edges: ", a.sum())
    print("==================================================================================")
    # input()

    break



# netgan = NetGAN(_N, rw_len, walk_generator= walker.walk, gpu_id=0, use_gumbel=True, disc_iters=3,
#                 W_down_discriminator_size=128, W_down_generator_size=128,
#                 l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
#                 generator_layers=[40], discriminator_layers=[30], temp_start=5, learning_rate=0.0003)
netgan = NetGAN(_N, rw_len, 
                walk_generator_1=fb_walker.walk, privacy_walk_generator_1=None, 
                walk_generator_2=None, privacy_walk_generator_2=None, 
                walk_generator_3=None, privacy_walk_generator_3=None, 
                gpu_id=0, use_gumbel=True, disc_iters=3, noise_dim=rw_len,
                W_down_discriminator_size=512, W_down_generator_size=512, batch_size=batch_size,
                l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                generator_layers=[60], discriminator_layers=[60], temp_start=5, learning_rate=0.0003,
                dir_name=opt.dir_name, legacy_generator=False)

stopping_criterion = "val"

assert stopping_criterion in ["val", "eo"], "Please set the desired stopping criterion."

if stopping_criterion == "val": # use val criterion for early stopping
    stopping = None
elif stopping_criterion == "eo":  #use eo criterion for early stopping
    stopping = 0.5 # set the target edge overlap here



#================
# Train the model
#================
# eval_every = 58000
# plot_every = 58000
# eval_every = 9999 
# plot_every = 9999
eval_every = 2000
plot_every = 2000
# eval_every = 500
# plot_every = 500
# eval_every = 20
# plot_every = 20

# log_dict = netgan.train(A_orig=_A_obs, val_ones=val_ones, val_zeros=val_zeros, stopping=stopping,
#                         eval_every=eval_every, plot_every=plot_every, max_patience=20, max_iters=200000)
log_dict = netgan.train(A_orig=train_graph, val_ones=val_ones, val_zeros=val_zeros, stopping=stopping,
                        eval_every=eval_every, plot_every=plot_every, max_patience=20, max_iters=20000,
                        model_name=opt.dir_name, continue_training=False, K=K, SMALLER_K=SMALLER_K, evaluate=False
                        , label=ground_truth_community)

print(log_dict.keys())
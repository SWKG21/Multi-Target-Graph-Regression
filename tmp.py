import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from preprocessing_baseline import *
from utils import *


path_to_data = '../data/'

embeddings = np.load(path_to_data + 'embeddings.npy')  # (#nodes, 13)
print ('node embeddings shape', embeddings.shape)

## WL relabeling
edgelists = os.listdir(path_to_data + 'edge_lists/')
edgelists.sort(key=natural_keys)

graphs = []
nodes = []
for idx, edgelist in enumerate(edgelists):
    g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist)
    graphs.append(g)
    for node in g.nodes():
        nodes.append(node)

assert len(nodes) == len(set(nodes))
labels = wl_relabeling(graphs, 3)
assert len(labels) == len(graphs)

# combine relabel to embedding, output
node_labels = np.zeros((embeddings.shape[0], 1), dtype=np.int32)
for i, g in enumerate(graphs):
    assert g.number_of_nodes() == len(labels[i])
    for j, node in enumerate(g.nodes()):
        node_labels[int(node), 0] = labels[i][j]

embeddings_relabel = np.concatenate([embeddings, node_labels], axis=1)
print ('node embedding shape after adding relabels', embeddings_relabel.shape)
scaler = MinMaxScaler()
embeddings_relabel = scaler.fit_transform(embeddings_relabel)
np.save(path_to_data + 'embeddings_relabel.npy', embeddings_relabel, allow_pickle=False)





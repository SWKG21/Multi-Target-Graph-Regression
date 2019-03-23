import os
from time import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils import *

# = = = = = = = = = = = = = = =

path_to_data = 'data/'

# 0-based index of the last row of the embedding matrix (for zero-padding)
pad_vec_idx = 1685894

# split to parts
parts = 4

# node2vec walks parameters
p = 0.5  # 1/p to go back
q = 3  # 1/q to go out
num_walks = 6
min_walk_length = 5
max_walk_length = 15

# node2vec embedding parameters
embedding_dim = 12
window_size = 5
node_embed_scale = 10

# maximum number of 'sentences' (walks) in each pseudo-document
max_doc_size = 90

# WL relabeling iterations
iterations = 5


# = = = = = = = = = = = = = = =

# sort by graph ID
edgelists = os.listdir(path_to_data + 'edge_lists/')
edgelists.sort(key=natural_keys)

### -------------------- node2vec sampling and embedding -------------------- ###

# generate documents and node embeddings by part because of memory limitation
start = 0
end = round(len(edgelists)/parts)
for i in range(1, parts):
    generate_docs_embeddings(path_to_data, i, edgelists[start:end], embedding_dim, window_size, num_walks, min_walk_length, max_walk_length, p, q, pad_vec_idx, max_doc_size)
    start = end
    end += round(len(edgelists)/parts)
generate_docs_embeddings(path_to_data, parts, edgelists[start:], embedding_dim, window_size, num_walks, min_walk_length, max_walk_length, p, q, pad_vec_idx, max_doc_size)

# combine parts of documents
concatenate_npy(path_to_data, 'documents', parts, 'documents_p0_5q3.npy')

# combine parts of node embeddings
concatenate_npy(path_to_data, 'node_embed', parts, 'node_embed_p0_5q3.npy')


### -------------------- WL relabeling -------------------- ###

# load graphs and nodes
graphs = []
nodes = []
for idx, edgelist in enumerate(edgelists):
    g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist)
    graphs.append(g)
    for node in g.nodes():
        nodes.append(node)

# wl relabeling
labels = wl_relabeling(graphs, iterations)

# assign labels to nodes in order
node_labels = np.zeros((len(nodes)+1, 1), dtype=np.int32)
for i, g in enumerate(graphs):
    assert g.number_of_nodes() == len(labels[i])
    for j, node in enumerate(g.nodes()):
        node_labels[int(node), 0] = labels[i][j]

# add relabels to attributes
attributes = np.load(path_to_data + 'embeddings.npy')  # (#nodes, 13)
print ('\nnode attributes shape', attributes.shape)
attributes_relabel = np.concatenate((attributes, node_labels), axis=1)
print ('node attributes shape after adding relabels', attributes_relabel.shape)


### -------------------- combine attributes and node embeddings -------------------- ###

# attributes scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
attributes_relabel = scaler.fit_transform(attributes_relabel)

# node mebeddings scaling
node_embeddings = np.load(path_to_data + 'node_embed_p0_5q3.npy')
node_embeddings = node_embeddings * node_embed_scale

# combine attributes and node embeddings
embeddings = np.concatenate((attributes_relabel, node_embeddings), axis=1)
np.save(path_to_data + 'embeddings_p0_5q3.npy', embeddings, allow_pickle=False)
print ('\nfinal embeddings shape', embeddings.shape)
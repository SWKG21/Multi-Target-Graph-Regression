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
p = 2  # 1/p to go back
q = 0.5  # 1/q to go out
num_walks = 6
min_walk_length = 10
max_walk_length = 12

# node2vec embedding parameters
embedding_dim = 12
window_size = 5
node_embed_scale = 10

# maximum number of 'sentences' (walks) in each pseudo-document
max_doc_size = 50

# WL relabeling iterations
iterations = 10


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
concatenate_npy(path_to_data, 'documents', parts, 'documents_p2q_5_wl10_em12.npy')

# combine parts of node embeddings
concatenate_npy(path_to_data, 'node_embed', parts, 'node_embed_p2q_5_wl10_em12.npy')


### -------------------- WL relabeling & extra node features -------------------- ###

# load graphs and nodes
graphs = []
nodes = []
degrees = []
degree_cens = []
closeness_cens = []
betweenness_cens = []
current_flow_closeness_cens = []
current_flow_betweenness_cens = []
appro_current_flow_betweenness_cens = []
cores = []
pageranks = []
dens = []
num_nodes = []
num_edges =[]
for idx, edgelist in enumerate(edgelists):
	g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist)
	graphs.append(g)
	for node in g.nodes():
		nodes.append(node)
	degrees.append(g.degree())
	degree_cens.append(nx.degree_centrality(g))
	closeness_cens.append(nx.closeness_centrality(g))
	betweenness_cens.append(nx.betweenness_centrality(g))
	current_flow_closeness_cens.append(nx.current_flow_closeness_centrality(g))
	current_flow_betweenness_cens.append(nx.current_flow_betweenness_centrality(g))
	appro_current_flow_betweenness_cens.append(nx.approximate_current_flow_betweenness_centrality(g))
	cores.append(nx.core_number(g))
	pageranks.append(nx.pagerank(g))
	dens.append(nx.density(g))
	num_nodes.append(nx.number_of_nodes(g))
	num_edges.append(nx.number_of_edges(g))

# wl relabeling
labels = wl_relabeling(graphs, iterations)

# assign labels and features to nodes in order
node_features = np.zeros((len(nodes)+1, 13), dtype=np.int32)
for i, g in enumerate(graphs):
	assert g.number_of_nodes() == len(labels[i])
	for j, node in enumerate(g.nodes()):
		node_features[int(node), 0] = labels[i][j]
		node_features[int(node), 1] = degrees[i][node]
		node_features[int(node), 2] = degree_cens[i][node]
		node_features[int(node), 3] = closeness_cens[i][node]
		node_features[int(node), 4] = betweenness_cens[i][node]
		node_features[int(node), 5] = current_flow_closeness_cens[i][node]
		node_features[int(node), 6] = current_flow_betweenness_cens[i][node]
		node_features[int(node), 7] = appro_current_flow_betweenness_cens[i][node]
		node_features[int(node), 8] = cores[i][node]
		node_features[int(node), 9] = pageranks[i][node]
		node_features[int(node), 10] = dens[i]
		node_features[int(node), 11] = num_nodes[i]
		node_features[int(node), 12] = num_edges[i]


# add relabels and features to attributes
attributes = np.load(path_to_data + 'embeddings.npy')  # (#nodes, 13)
print ('\nnode attributes shape', attributes.shape)
attributes_added = np.concatenate((attributes, node_features), axis=1)  # (#nodes, 23)
print ('node attributes shape after adding relabels and features', attributes_added.shape)


### -------------------- combine attributes and node embeddings -------------------- ###

# attributes scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
attributes_added = scaler.fit_transform(attributes_added)

# node mebeddings scaling
node_embeddings = np.load(path_to_data + 'node_embed_p2q_5_wl10_em12.npy')
node_embeddings = node_embeddings * node_embed_scale

# combine attributes and node embeddings
embeddings = np.concatenate((attributes_added, node_embeddings), axis=1)
np.save(path_to_data + 'embeddings_p2q_5_wl10_em12.npy', embeddings, allow_pickle=False)
print ('\nfinal embeddings shape', embeddings.shape)
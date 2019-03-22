from collections import defaultdict
import copy

import keras.backend as K
from keras.layers import Bidirectional, GRU, CuDNNGRU, LSTM, CuDNNLSTM

import os
import re
import random
import numpy as np
import networkx as nx
from node2vec import Node2Vec


# 'atoi' and 'natural_keys' taken from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def random_walk(graph, node, walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        walk.append(random.choice(list(neighbors)))
    return walk


def generate_walks(graph, num_walks, walk_length):
    '''
    samples num_walks walks of length walk_length+1 from each node of graph
    '''
    graph_nodes = graph.nodes()
    n_nodes = len(graph_nodes)
    walks = []
    for i in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for j in range(n_nodes):
            walk = random_walk(graph, nodes[j], walk_length)
            walks.append(walk)
    return walks


def generate_random_walks(graph, dimensions, window, num_walks, min_walk_length,
                          max_walk_length, p, q, pad_vec_idx):
    '''
    generate random walks by node2vec
    '''
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=max_walk_length,
                        num_walks=num_walks, workers=2, p=p, q=q, quiet=True)
    walks = node2vec.walks
    walk_length = np.random.randint(
        min_walk_length, max_walk_length+1, size=len(walks))
    # padding for walk
    walks = [w[:walk_length[idx]] + [pad_vec_idx] *
             (max_walk_length-walk_length[idx]) for idx, w in enumerate(walks)]
    model = node2vec.fit(window=window, min_count=1, batch_words=4)
    return walks, model


def preprocessing(path_to_data, edgelists, part, dimensions, window, num_walks,
                  min_walk_length, max_walk_length, p, q, pad_vec_idx, max_doc_size, show=25):
    docs = []
    embed = []
    for idx, edgelist in enumerate(edgelists):
        # construct graph from edgelist
        g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist)
        # create the pseudo-document representation of the graph
        doc, model = generate_random_walks(g, dimensions, window, num_walks,
                                           min_walk_length, max_walk_length, p, q, pad_vec_idx)
        docs.append(doc)
        min_idx = min(map(int, model.wv.index2word))
        length = len(model.wv.index2word)
        for i in range(length):
            embed.append(model[str(min_idx+i)])

        if idx % round(len(edgelists)/show) == 0:
            print(idx)

    print('documents generated')

    # truncation-padding at the document level, i.e., adding or removing entire 'sentences'
    docs = [d+[[pad_vec_idx]*max_walk_length] * (max_doc_size-len(d))
            if len(d) < max_doc_size else d[:max_doc_size] for d in docs]

    docs = np.array(docs).astype('int')
    print('documents '+part+' array shape:', docs.shape)
    embed = np.array(embed).astype('float32')
    print('node_embed '+part+' array shape:', embed.shape)

    np.save(path_to_data + 'documents'+part+'.npy', docs, allow_pickle=False)
    np.save(path_to_data + 'node_embed'+part+'.npy', embed, allow_pickle=False)
    print('part ' + part + ' saved')


def concatenate_npy(path_to_data, name, save_name):
    '''
    concatenate two npy documents
    '''
    doc1 = np.load(path_to_data + name + '1.npy')
    doc2 = np.load(path_to_data + name + '2.npy')
    doc = np.concatenate((doc1, doc2), axis=0)
    print(name + ' array shape:', doc.shape)
    
    np.save(path_to_data + save_name, doc, allow_pickle=False)
    os.remove(path_to_data + name + '1.npy')
    os.remove(path_to_data + name + '2.npy')


def embed_weight(path_to_data, embed_name, node_embed_name, embed_max, embed_min, node_max, node_min):
    '''
    adjust embedding weights in two embedding documents
    '''
    embeddings = np.load(path_to_data + embed_name)[:-1]
    node_embed = np.load(path_to_data + node_embed_name)
    for i in range(embeddings.shape[1]):
        max_embed = max(embeddings[:, i])
        min_embed = min(embeddings[:, i])
        embeddings[:, i] = (embed_max - embed_min)/(max_embed - min_embed) * \
                           (embeddings[:, i] - min_embed) + embed_min
    max_node = np.max(node_embed)
    min_node = np.min(node_embed)
    node_embed = (node_max - node_min)/(max_node - min_node) * (node_embed - min_node) + node_min
    return np.concatenate((embeddings, node_embed), axis=1)


def wl_relabeling(graphs, h):

	N = len(graphs)

	labels = {}
	label_lookup = {}
	label_counter = 0

	for G in graphs:
		for node in G.nodes():
			G.node[node]['label'] = G.degree(node)

	orig_graph_map = {it: {i: defaultdict(lambda: 0) for i in range(N)} for it in range(-1, h)}

    # initial labeling
	ind = 0
	for G in graphs:
		labels[ind] = np.zeros(G.number_of_nodes(), dtype=np.int32)
		node2index = {}
		for node in G.nodes():
		    node2index[node] = len(node2index)
		    
		for node in G.nodes():
		    label = G.node[node]['label']
		    if label not in label_lookup:
		        label_lookup[label] = len(label_lookup)

		    labels[ind][node2index[node]] = label_lookup[label]
		    orig_graph_map[-1][ind][label] = orig_graph_map[-1][ind].get(label, 0) + 1
		
		ind += 1
		
	compressed_labels = copy.deepcopy(labels)

    # WL iterations
	for it in range(h):
		unique_labels_per_h = set()
		label_lookup = {}
		ind = 0
		for G in graphs:
			node2index = {}
			for node in G.nodes():
				node2index[node] = len(node2index)
				
			for node in G.nodes():
				node_label = tuple([labels[ind][node2index[node]]])
				neighbors = G.neighbors(node)
				neighbors = [n for n in neighbors]  # transform dict_keyiterator to list
				if len(neighbors) > 0:
					neighbors_label = tuple([labels[ind][node2index[neigh]] for neigh in neighbors])
					node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
				if node_label not in label_lookup:
					label_lookup[node_label] = len(label_lookup)
					
				compressed_labels[ind][node2index[node]] = label_lookup[node_label]
				orig_graph_map[it][ind][node_label] = orig_graph_map[it][ind].get(node_label, 0) + 1
		        
			ind +=1

		labels = copy.deepcopy(compressed_labels)

	return labels


def dot_product(x, kernel):
    """
    https://github.com/richliao/textClassifier/issues/13#issuecomment-377323318
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


def bidir_gru(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh', 
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)


def bidir_lstm(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with LSTM units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNLSTM(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(LSTM(units=n_units,
                                 activation='tanh', 
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)
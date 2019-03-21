import numpy as np
from collections import defaultdict
import copy

import keras.backend as K
from keras.layers import Bidirectional, GRU, CuDNNGRU, LSTM, CuDNNLSTM


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
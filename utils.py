import numpy as np
from collections import defaultdict
import copy


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
		labels[ind] = np.zeros(G.number_of_nodes(), dtype = np.int32)
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
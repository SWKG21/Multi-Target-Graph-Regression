import copy
import os
import re
import random
from collections import defaultdict

import numpy as np
import networkx as nx
from node2vec import Node2Vec

import keras.backend as K
from keras.layers import Bidirectional, GRU, CuDNNGRU, LSTM, CuDNNLSTM


def atoi(text):
    """
        if text is digit, return an integer;
        otherwise, return original text
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
        extract possible integers in text
        return a list
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def random_walk(graph, node, walk_length):
    """
        starting from node in graph, generate a walk with length walk_length+1 
        by randomly choosing one neighbor. 
    """
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        walk.append(random.choice(list(neighbors)))
    return walk


def generate_walks(graph, num_walks, walk_length):
    """
        for each node of graph, generate num_walks walks with length walk_length+1
        by using random_walk function
    """
    graph_nodes = graph.nodes()
    n_nodes = len(graph_nodes)
    walks = []
    for i in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for j in range(n_nodes):
            walk = random_walk(graph, nodes[j], walk_length)
            walks.append(walk)
    return walks


def generate_node2vec_walks(graph, embedding_dim, window_size, num_walks, min_walk_length, max_walk_length, p, q, pad_vec_idx):
    """
        generate random walks by using node2vec, walk length can be not fixing.
        parameters:
            embedding_dim: node embedding dimension;
            min_walk_length, max_walk_length: control the range of random walk length;
            p, q: control the probability of random walk direction;
            pad_vec_idx: node index for padding.
    """
    node2vec = Node2Vec(graph, dimensions=embedding_dim, walk_length=max_walk_length, num_walks=num_walks, workers=2, p=p, q=q, quiet=True)
    walks = node2vec.walks
    # for each walk, randomly set their length
    walks_length = np.random.randint(min_walk_length, max_walk_length+1, size=len(walks))
    # padding for walk
    walks = [w[:walks_length[idx]] + [pad_vec_idx] * (max_walk_length-walks_length[idx]) for idx, w in enumerate(walks)]
    model = node2vec.fit(window=window_size, min_count=1, batch_words=4)
    return walks, model


def generate_docs_embeddings(path_to_data, part, edgelists, embedding_dim, window_size, num_walks, min_walk_length, max_walk_length, p, q, pad_vec_idx, max_doc_size):
    """
        generate documents and node embeddings from graphs by using node2vec; because of memory limitation, generation by part.
        parameters:
            part: integer, part-th part of generation;
            edgelists: graphs for part-th part;
            embedding_dim: node embedding dimension;
            min_walk_length, max_walk_length: control the range of random walk length;
            p, q: control the probability of random walk direction;
            pad_vec_idx: node index for padding;
            max_doc_size: maximum number of 'sentences' (walks) in each pseudo-document.
    """
    docs = []
    embed = []
    for idx, edgelist in enumerate(edgelists):
        # construct graph from edgelist
        g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist)
        # create the pseudo-document representation of the graph
        doc, model = generate_node2vec_walks(g, embedding_dim, window_size, num_walks, min_walk_length, max_walk_length, p, q, pad_vec_idx)
        docs.append(doc)
        # get node embeddings
        min_idx = min(map(int, model.wv.index2word))
        length = len(model.wv.index2word)
        for i in range(length):
            embed.append(model[str(min_idx+i)])
        # print out process
        if idx % round(len(edgelists)/25) == 0:
            print(idx, 'finished')

    print('\ndocuments generated')

    # truncation-padding at the document level, i.e., adding or removing entire 'sentences'
    docs = [d + [[pad_vec_idx]*max_walk_length] * (max_doc_size-len(d)) if len(d) < max_doc_size else d[:max_doc_size] for d in docs]
    # transform to np.array
    docs = np.array(docs).astype('int')
    print('documents' + str(part) + ' array shape:', docs.shape)
    embed = np.array(embed).astype('float32')
    print('node_embed' + str(part) + ' array shape:', embed.shape)
    # save
    np.save(path_to_data + 'documents' + str(part) + '.npy', docs, allow_pickle=False)
    np.save(path_to_data + 'node_embed' + str(part) + '.npy', embed, allow_pickle=False)
    print('part' + str(part) + ' saved')


def concatenate_npy(path_to_data, filename, parts, save_name):
    """
        concatenate parts and save (documents / node embeddings);
        add 0-based index in the last row of the embedding matrix.
    """
    docs = []
    for i in range(1, parts+1):
        doc_n = np.load(path_to_data + filename + str(i) + '.npy')
        docs.append(doc_n)
        # remove temporary split parts
        os.remove(path_to_data + filename + str(i) + '.npy')
    doc = np.concatenate(docs, axis=0)
    # add 0-based index in the last row of the embedding matrix
    if filename == 'node_embed':
        doc = np.concatenate((doc, np.zeros([1, doc.shape[1]])), axis=0)
    print('\nconcatenated array shape:', doc.shape)
    np.save(path_to_data + save_name, doc, allow_pickle=False)


def wl_relabeling(graphs, h):
    """
        Weisfeiler Lehman relabeling given graphs and number of iterations.
    """
    N = len(graphs)
    labels = {}
    label_lookup = {}
    label_counter = 0

    # degree as initial node label
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


def generate_contexutal_docs(path_to_data, filename, window_size):
    """
        For each doc in a sequence of docs, find its neighbor docs in a window;
        expand original 3D array to 4D to store each doc and its neighbor docs.
    """
    docs = np.load(path_to_data + filename)
    new_docs = np.zeros((docs.shape[0], docs.shape[1], 1+window_size, docs.shape[2]))
    print ('original docs shape', docs.shape)

    half = int(window_size/2)
    for i in range(docs.shape[0]):
        for j in range(docs.shape[1]):
            if j < half:
                start = 0
                end = window_size+1
            elif j + half > docs.shape[1] - 1:
                start = docs.shape[1] - 1 - window_size
                end = docs.shape[1]
            else:
                start = j - half
                end = j + half
            
            new_docs[i, j, 0, :] = docs[i, j, :]
            for k in range(start, j):
                new_docs[i, j, 1+k-start, :] = docs[i, k, :]
            for k in range(j+1, end):
                new_docs[i, j, k-start, :] = docs[i, k, :]
        # print out process
        if i % round(docs.shape[0]/10) == 0:
            print (i, 'finished')
    # save
    np.save(path_to_data + 'contextual'+ str(window_size) +'_' + filename, new_docs, allow_pickle=False)


def data_load(path_to_data, docs_file, tgt):
    """
        create training and validation set.
    """
    # load train indices
    with open(path_to_data + 'train_idxs.txt', 'r') as file:
        train_idxs = file.read().splitlines()
    train_idxs = [int(elt) for elt in train_idxs]

    # create train and val
    np.random.seed(12219)
    idxs_select_train = np.random.choice(range(len(train_idxs)),size=int(len(train_idxs)*0.80),replace=False)
    idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)
    
    # split train and val indices
    train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
    val_idxs = [train_idxs[elt] for elt in idxs_select_val]
    
    # split train and val docs
    docs = np.load(path_to_data + docs_file)
    docs_train = docs[train_idxs_new,:,:]
    docs_val = docs[val_idxs,:,:]
    
    # load targets
    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()
    # split targets into train and val
    target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
    target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')

    return docs_train, docs_val, target_train, target_val


def dot_product(x, kernel):
    """
        Wrapper for dot product operation
        Args:
            x (): input
            kernel (): weights
        Returns:
    """
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)


def bidir_gru(my_seq,n_units,is_GPU):
    '''
        just a convenient wrapper for bidirectional RNN with GRU units
        enables CUDA acceleration on GPU
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
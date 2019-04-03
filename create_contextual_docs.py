import sys
import json
import numpy as np


# = = = = = = = = = = = = = = =

is_GPU = False
save_weights = True
save_history = True

path_root = '..'
path_to_code = path_root + '/code/'
path_to_data = path_root + '/data/'

sys.path.insert(0, path_to_code)

# = = = = = = = = = = = = = = =

window_size = 6  # should be even

# = = = = = data loading = = = = =

docs = np.load(path_to_data + 'documents_p2q_5_new.npy')
new_docs = np.zeros((docs.shape[0], docs.shape[1], 1+window_size, docs.shape[2]))
print (docs.shape)

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
    if i % 10000 == 0:
        print (i, 'finished')

np.save(path_to_data + 'contextual'+ str(window_size) +'_documents_p2q_5_new.npy', new_docs, allow_pickle=False)
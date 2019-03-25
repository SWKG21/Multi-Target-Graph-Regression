import sys
import json
import numpy as np

from keras.backend import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, TimeDistributed, Dense, Add, add, Lambda, Reshape

from utils import *
from AttentionWithContext import AttentionWithContext
from StructuredSelfAttentive import StructuredSelfAttentive
from AttentionWithMultiContext import AttentionWithMultiContext
from SkipConnection import SkipConnection
from AttentionWithGlobalContext import AttentionWithGlobalContext


"""
    sentence encoder: AttentionWithContext (first s in file name), add SkipConnection
    document encoder: AttentionWithMultiContext, AttentionWithContext (second s in file name) for u
"""


# = = = = = = = = = = = = = = =

is_GPU = False
save_weights = True
save_history = True

path_root = '..'
path_to_code = path_root + '/code/'
path_to_data = path_root + '/data/'

sys.path.insert(0, path_to_code)

# = = = = = = = = = = = = = = =

# = = = = = hyper-parameters = = = = =

n_units = 30
gc_n_units = 40
da = 15
r = 10
drop_rate = 0.3
batch_size = 128
nb_epochs = 100
my_optimizer = 'adam'
my_patience = 4


# = = = = = data loading = = = = =

docs = np.load(path_to_data + 'documents_p2q_5_50.npy')
embeddings = np.load(path_to_data + 'embeddings_p2q_5.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()
    
train_idxs = [int(elt) for elt in train_idxs]

# create validation set
np.random.seed(12219)
idxs_select_train = np.random.choice(range(len(train_idxs)),size=int(len(train_idxs)*0.80),replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)

train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]

docs_train = docs[train_idxs_new,:,:]
docs_val = docs[val_idxs,:,:]

tgt = 0

with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
    target = file.read().splitlines()

target_train = np.array([target[elt] for elt in idxs_select_train]).astype('float')
target_val = np.array([target[elt] for elt in idxs_select_val]).astype('float')

print('data loaded')
print (docs_train.shape)
# = = = = = defining architecture = = = = =

sent_ints = Input(shape=(docs_train.shape[2],))

sent_wv = Embedding(input_dim=embeddings.shape[0],
                    output_dim=embeddings.shape[1],
                    weights=[embeddings],
                    input_length=docs_train.shape[2],
                    trainable=False,
                    )(sent_ints)

## HAN sent encoder for context
sent_wv_dr = Dropout(drop_rate)(sent_wv)
sent_wa = bidir_gru(sent_wv_dr, n_units, is_GPU)
sent_att_vec, word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa)
sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)
sent_encoder = Model(sent_ints, sent_att_vec_dr)

# encode all sentences for context
context_doc_ints = Input(shape=(docs_train.shape[1], docs_train.shape[2],))
sent_att_vecs_dr = TimeDistributed(sent_encoder)(context_doc_ints)
context_encoder = Model(context_doc_ints, sent_att_vecs_dr)


## sentence encoder with global context
context_vecs = context_encoder(context_doc_ints)
gc_sent_wv_dr = Dropout(drop_rate)(sent_wv)
gc_sent_wa = bidir_gru(gc_sent_wv_dr, n_units, is_GPU)
gc_sent_att_vec, gc_word_att_coeffs = AttentionWithGlobalContext(return_coefficients=True)([gc_sent_wa, context_vecs])
gc_sent_att_vec_dr = Dropout(drop_rate)(gc_sent_att_vec)
gc_sent_encoder = Model(sent_ints, gc_sent_att_vec_dr)

## main structure
doc_ints = Input(shape=(docs_train.shape[1], docs_train.shape[2],))
gc_sent_att_vecs_dr = TimeDistributed(gc_sent_encoder)(doc_ints)
doc_sa = bidir_gru(gc_sent_att_vecs_dr, n_units, is_GPU)
doc_att_vec, sent_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_sa)
doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

preds = Dense(units=1)(doc_att_vec_dr)
model = Model(doc_ints, preds)

model.compile(loss='mean_squared_error', optimizer=my_optimizer, metrics=['mae'])

print('model compiled')

# = = = = = training = = = = =

early_stopping = EarlyStopping(monitor='val_loss',
                                patience=my_patience,
                                mode='min')

# save model corresponding to best epoch
checkpointer = ModelCheckpoint(filepath=path_to_data + 'model_ss' + str(tgt), 
                                verbose=1, 
                                save_best_only=True,
                                save_weights_only=True)

if save_weights:
    my_callbacks = [early_stopping, checkpointer]
else:
    my_callbacks = [early_stopping]

model.fit(docs_train, 
            target_train,
            batch_size = batch_size,
            epochs = nb_epochs,
            validation_data = (docs_val,target_val),
            callbacks = my_callbacks)

hist = model.history.history

if save_history:
    with open(path_to_data + 'model_history_ss' + str(tgt) + '.json', 'w') as file:
        json.dump(hist, file, sort_keys=False, indent=4)

print('* * * * * * * target',tgt,'done * * * * * * *')    
    

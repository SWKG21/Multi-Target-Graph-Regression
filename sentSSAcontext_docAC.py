import sys
import json
import numpy as np

from keras.backend import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, TimeDistributed, Dense, Lambda, Reshape, Concatenate, LeakyReLU

from utils import *
from custom_layers import *


"""
    replace u in sentence encoder;
    sentence encoder for u: StructuredSelfAttentive;
    sentence encoder: SentContextAttention; add SkipConnection;
    document encoder: AttentionWithContext;
    removing the sigmoid activation function in the last layer;
    add 2 LeakyReLU before last layer;
"""

# = = = = = = = = = = = = = = =

is_GPU = True
save_weights = True
save_history = True

path_root = ''
path_to_code = path_root + '/code/'
path_to_data = path_root + 'data/'

sys.path.insert(0, path_to_code)

# = = = = = = = = = = = = = = =

embeds_file = 'embeddings_final.npy'
docs_file = 'contextual6_documents_final.npy'

# = = = = = hyper-parameters = = = = =

con_n_units = 60
da = 15
r = 10
sent_n_units = 80
doc_n_units = 100
drop_rate = 0.3
batch_size = 256
nb_epochs = 100
my_optimizer = 'adam'
my_patience = 4
tgt = 0

# = = = = = data loading = = = = =

embeddings = np.load(path_to_data + embeds_file)
docs_train, docs_val, target_train, target_val = data_load(path_to_data, docs_file, tgt)
print('data loaded')

# = = = = = defining architecture = = = = =

### SSA sent encoder for context
con_sent_ints = Input(shape=(docs_train.shape[3],))
con_sent_wv = Embedding(input_dim=embeddings.shape[0],
                    output_dim=embeddings.shape[1],
                    weights=[embeddings],
                    input_length=docs_train.shape[3],
                    trainable=False,
                    )(con_sent_ints)
con_sent_wv_dr = Dropout(drop_rate)(con_sent_wv)
con_sent_wa = bidir_gru(con_sent_wv_dr, con_n_units, is_GPU)
con_sent_att_vec = StructuredSelfAttentive(da=da, r=r)(con_sent_wa)
con_sent_att_vec_dr = Dropout(drop_rate)(con_sent_att_vec)
con_sent_encoder = Model(con_sent_ints, con_sent_att_vec_dr)

### context encoder with SSA sent encoders
con_ints = Input(shape=(docs_train.shape[2]-1, docs_train.shape[3],))
con_sent_att_vecs_dr = TimeDistributed(con_sent_encoder)(con_ints)
context_encoder = Model(con_ints, con_sent_att_vecs_dr)

### sentence encoder with context
sent_context_ints = Input(shape=(docs_train.shape[2], docs_train.shape[3],))
sent_ints = Lambda(lambda x: x[:, 0, :])(sent_context_ints)
context_ints = Lambda(lambda x: x[:, 1:, :])(sent_context_ints)
sent_wv = Embedding(input_dim=embeddings.shape[0],
                    output_dim=embeddings.shape[1],
                    weights=[embeddings],
                    input_length=docs_train.shape[3],
                    trainable=False,
                    )(sent_ints)
sent_wv_dr = Dropout(drop_rate)(sent_wv)
sent_wa = bidir_gru(sent_wv_dr, sent_n_units, is_GPU)
context_vecs = context_encoder(context_ints)
sent_att_vec = SentContextAttention()([sent_wa, context_vecs])
sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)
sent_encoder = Model(sent_context_ints, sent_att_vec_dr)
# # skip connection
# sent_added = SkipConnection()([sent_att_vec_dr, sent_wv_dr])
# sent_encoder = Model(sent_context_ints, sent_added)

### doc encoder
doc_ints = Input(shape=(docs_train.shape[1], docs_train.shape[2], docs_train.shape[3],))
sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
doc_sa = bidir_gru(sent_att_vecs_dr, doc_n_units, is_GPU)
doc_att_vec = AttentionWithContext()(doc_sa)
doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

doc_att_vec_dr = LeakyReLU(alpha=0.01)(doc_att_vec_dr)
doc_att_vec_dr = LeakyReLU(alpha=0.01)(doc_att_vec_dr)
preds = Dense(units=1)(doc_att_vec_dr)
model = Model(doc_ints, preds)

model.compile(loss='mean_squared_error', optimizer=my_optimizer, metrics=['mae'])

print('model compiled')

# = = = = = training = = = = =

early_stopping = EarlyStopping(monitor='val_loss',
                                patience=my_patience,
                                mode='min')

# save model corresponding to best epoch
checkpointer = ModelCheckpoint(filepath=path_to_data + 'model_' + str(tgt), 
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
    with open(path_to_data + 'model_history_' + str(tgt) + '.json', 'w') as file:
        json.dump(hist, file, sort_keys=False, indent=4)

print('* * * * * * * target',tgt,'done * * * * * * *')    
    

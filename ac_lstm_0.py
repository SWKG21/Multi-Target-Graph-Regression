import sys
import json
import numpy as np

import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, TimeDistributed, Dense, Lambda, LeakyReLU

from utils import *
from custom_layers import *


"""
    replace u in document encoder;
    sentence encoder: AttentionWithContext; add SkipConnection;
    sentence encoder for u: BiLSTM; add SkipConnection;
    document encoder: DocStructuredAttention; add SkipConnection;
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
docs_file = 'documents_final.npy'

# = = = = = hyper-parameters = = = = =

sent_n_units = 80
u_n_units = 60
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

sent_ints = Input(shape=(docs_train.shape[2],))

sent_wv = Embedding(input_dim=embeddings.shape[0],
                    output_dim=embeddings.shape[1],
                    weights=[embeddings],
                    input_length=docs_train.shape[2],
                    trainable=False,
                    )(sent_ints)

### AC sent encoder
sent_wv_dr = Dropout(drop_rate)(sent_wv)
sent_wa = bidir_gru(sent_wv_dr, sent_n_units, is_GPU)
sent_att_vec = AttentionWithContext()(sent_wa)
sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)
# skip connection
sent_added = SkipConnection()([sent_att_vec_dr, sent_wv_dr])
sent_encoder = Model(sent_ints, sent_added)


### LSTM encoder for u
u_sent_wv_dr = Dropout(drop_rate)(sent_wv)
u_sent_wa = bidir_lstm(u_sent_wv_dr, u_n_units, is_GPU)
u_sent_att_vec = Lambda(lambda x: K.mean(x, axis=1))(u_sent_wa)
u_sent_att_vec_dr = Dropout(drop_rate)(u_sent_att_vec)
# skip connection
u_sent_added = SkipConnection()([u_sent_att_vec_dr, u_sent_wv_dr])
u_sent_encoder = Model(sent_ints, u_sent_added)


### doc encoder
doc_ints = Input(shape=(docs_train.shape[1], docs_train.shape[2],))
# AC sentence encoders
sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
doc_sa = bidir_gru(sent_att_vecs_dr, doc_n_units, is_GPU)
# LSTM encoders for u
u_sent_att_vecs_dr = TimeDistributed(u_sent_encoder)(doc_ints)
u_doc_sa = bidir_gru(u_sent_att_vecs_dr, doc_n_units, is_GPU)
# combine
doc_att_vec = DocStructuredAttention()([doc_sa, u_doc_sa])
doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)
# skip connection
doc_added = SkipConnection()([doc_att_vec_dr, sent_att_vecs_dr])

doc_added = LeakyReLU(alpha=0.01)(doc_added)
doc_added = LeakyReLU(alpha=0.01)(doc_added)
preds = Dense(units=1)(doc_added)
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
    

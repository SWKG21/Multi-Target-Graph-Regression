import sys
import json
import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, TimeDistributed, Dense

from utils import *
from AttentionWithContext import AttentionWithContext
from StructuredSelfAttentive import StructuredSelfAttentive
from DocStructuredAttention import DocStructuredAttention
from SkipConnection import SkipConnection


# = = = = = = = = = = = = = = =

is_GPU = True

path_root = ''
path_to_data = path_root + 'data/'
path_to_code = path_root + '/code/'
sys.path.insert(0, path_to_code)

# = = = = = = = = = = = = = = =

docs = np.load(path_to_data + 'documents_p2q_5_new.npy')
embeddings = np.load(path_to_data + 'embeddings_p2q_5_new.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()

with open(path_to_data + 'test_idxs.txt', 'r') as file:
    test_idxs = file.read().splitlines()

train_idxs = [int(elt) for elt in train_idxs]
test_idxs = [int(elt) for elt in test_idxs]

docs_test = docs[train_idxs,:,:]

# = = = = = TRAINING RESULTS = = = = = 

tgt = 0
    
print('* * * * * * *',tgt,'* * * * * * *')

with open(path_to_data + '/model_history_sm_2l2l_' + str(tgt) + '.json', 'r') as file:
    hist = json.load(file)

val_mse = hist['val_loss']
val_mae = hist['val_mean_absolute_error']

min_val_mse = min(val_mse)
min_val_mae = min(val_mae)

best_epoch = val_mse.index(min_val_mse) + 1

print('best epoch:',best_epoch)
print('best val MSE',round(min_val_mse,3))
print('best val MAE',round(min_val_mae,3))

# = = = = = PREDICTIONS = = = = =     

# relevant hyper-parameters
n_units = 60
mc_n_units = 100
da = 15
r = 10
drop_rate = 0 # prediction mode

sent_ints = Input(shape=(docs_test.shape[2],))

sent_wv = Embedding(input_dim=embeddings.shape[0],
                    output_dim=embeddings.shape[1],
                    weights=[embeddings],
                    input_length=docs_test.shape[2],
                    trainable=False,
                    )(sent_ints)

## HAN sent encoder
sent_wv_dr = Dropout(drop_rate)(sent_wv)
sent_wa = bidir_gru(sent_wv_dr, n_units, is_GPU)
sent_wa = bidir_gru(sent_wa, n_units, is_GPU)
sent_att_vec, word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa)
sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)
# skip connection
sent_added = SkipConnection()([sent_att_vec_dr, sent_wv_dr])
sent_encoder = Model(sent_ints, sent_added)

## structured self-attentive
mc_sent_wv_dr = Dropout(drop_rate)(sent_wv)
mc_sent_wa = bidir_lstm(mc_sent_wv_dr, mc_n_units, is_GPU)
mc_sent_wa = bidir_lstm(mc_sent_wa, mc_n_units, is_GPU)
mc_sent_att_vec, mc_word_att_coeffs = StructuredSelfAttentive(da=da, r=r, return_coefficients=True)(mc_sent_wa)
mc_sent_att_vec_dr = Dropout(drop_rate)(mc_sent_att_vec)
# skip connection
mc_sent_added = SkipConnection()([mc_sent_att_vec_dr, mc_sent_wv_dr])
mc_sent_encoder = Model(sent_ints, mc_sent_added)

## doc encoder: combine context and target
doc_ints = Input(shape=(docs_test.shape[1], docs_test.shape[2],))
# sentence encoder
sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
doc_sa = bidir_gru(sent_att_vecs_dr, n_units, is_GPU)
doc_sa = bidir_gru(doc_sa, n_units, is_GPU)
# context
mc_sent_att_vecs_dr = TimeDistributed(mc_sent_encoder)(doc_ints)
mc_doc_sa = bidir_gru(mc_sent_att_vecs_dr, n_units, is_GPU)
mc_doc_sa = bidir_gru(mc_doc_sa, n_units, is_GPU)

doc_att_vec, sent_att_coeffs = DocStructuredAttention(return_coefficients=True)([doc_sa, mc_doc_sa])
doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

preds = Dense(units=1)(doc_att_vec_dr)
model = Model(doc_ints, preds)

model.load_weights(path_to_data + 'model_sm_2l2l_' + str(tgt))

preds = model.predict(docs_test).tolist()
preds = [pred[0] for pred in preds]

with open(path_to_data + 'predictions_'+ str(tgt) +'.txt', 'w') as file:
    for pred in preds:
        pred = format(pred, '.7f')
        file.write(pred + '\n')

import sys
import json
import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, TimeDistributed, Dense, LeakyReLU

from utils import *
from custom_layers import *


# = = = = = = = = = = = = = = =

is_GPU = True

path_root = ''
path_to_data = path_root + 'data/'
path_to_code = path_root + '/code/'
sys.path.insert(0, path_to_code)

# = = = = = = = = = = = = = = =

# = = = = = = = = = = = = = = =

docs = np.load(path_to_data + 'documents_final.npy')

with open(path_to_data + 'test_idxs.txt', 'r') as file:
    test_idxs = file.read().splitlines()
test_idxs = [int(elt) for elt in test_idxs]

docs_test = docs[test_idxs,:,:]

# = = = = = TRAINING RESULTS = = = = = 

for tgt in range(4):
    
    print('* * * * * * *',tgt,'* * * * * * *')
    
    with open(path_to_data + '/model_history_' + str(tgt) + '.json', 'r') as file:
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

all_preds_han = []

for tgt in range(4):
    
    print('* * * * * * *',tgt,'* * * * * * *')
    
    # * * * HAN * * * 
    
    # relevant hyper-parameters
    sent_n_units = 80
    doc_n_units = 100
    drop_rate = 0 # prediction mode
    
    sent_ints = Input(shape=(docs_test.shape[2],))

    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=docs_train.shape[2],
                        trainable=False,
                        )(sent_ints)

    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    sent_wa = bidir_gru(sent_wv_dr, sent_n_units, is_GPU)
    sent_att_vec = AttentionWithContext()(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)                      
    # skip connection
    sent_added = SkipConnection()([sent_att_vec_dr, sent_wv_dr])
    sent_encoder = Model(sent_ints, sent_added)

    doc_ints = Input(shape=(docs_test.shape[1],docs_test.shape[2],))
    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = bidir_gru(sent_att_vecs_dr, doc_n_units, is_GPU)
    doc_att_vec = AttentionWithContext()(doc_sa)
    doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)
    # skip connection
    doc_added = SkipConnection()([doc_att_vec_dr, sent_att_vecs_dr])

    doc_added = LeakyReLU(alpha=0.01)(doc_added)
    doc_added = LeakyReLU(alpha=0.01)(doc_added)
    preds = Dense(units=1)(doc_added)
    
    model = Model(doc_ints, preds)
    
    model.load_weights(path_to_data + 'model_' + str(tgt))
    
    all_preds_han.append(model.predict(docs_test).tolist())

# flatten
all_preds_han = [elt[0] for sublist in all_preds_han for elt in sublist]

with open(path_to_data + 'predictions_han.txt', 'w') as file:
    file.write('id,pred\n')
    for idx,pred in enumerate(all_preds_han):
        pred = format(pred, '.7f')
        file.write(str(idx) + ',' + pred + '\n')

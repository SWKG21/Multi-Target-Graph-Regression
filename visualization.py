import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path_to_data = '../data/'
test_pred = 'pred_han_sc.txt'
train_pred = 'pred_train_sc.txt'
val_pred = 'pred_val_sc.txt'

###--------------------------- load data -----------------------------###

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()
    
train_idxs = [int(elt) for elt in train_idxs]

# create validation set
np.random.seed(12219)
idxs_select_train = np.random.choice(range(len(train_idxs)),size=int(len(train_idxs)*0.80),replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)

train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]


###------------------ draw train and val distribution -------------------###

df_target_train = pd.DataFrame()
df_target_val = pd.DataFrame()
for tgt in range(4):
    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()
    target = [float(elt) for elt in target]
    target_train = [target[elt] for elt in idxs_select_train]
    target_val = [target[elt] for elt in idxs_select_val]
    df_target_train[tgt] = pd.to_numeric(target_train)
    df_target_val[tgt] = pd.to_numeric(target_val)
    
    print ('\ntarget '+str(tgt)+' train max', df_target_train[tgt].max())
    print ('target '+str(tgt)+' train mean', df_target_train[tgt].mean())
    print ('target '+str(tgt)+' train min', df_target_train[tgt].min())
    print ('target '+str(tgt)+' val max', df_target_val[tgt].max())
    print ('target '+str(tgt)+' val mean', df_target_val[tgt].mean())
    print ('target '+str(tgt)+' val min', df_target_val[tgt].min())
    
    plt.figure(figsize=(12, 6))
    plt.hist(df_target_train[tgt], bins=500)
    plt.title('Distribution of target '+str(tgt)+' values (train)')
    plt.savefig('plot/target'+str(tgt)+'_train.png')

    plt.figure(figsize=(12, 6))
    plt.hist(df_target_val[tgt], bins=500)
    plt.title('Distribution of target '+str(tgt)+' values (val)')
    plt.savefig('plot/target'+str(tgt)+'_val.png')


###------------------ draw prediction and error distribution -------------------###

preds = pd.read_csv(path_to_data + test_pred, index_col=0)
preds = pd.DataFrame(preds.values.reshape(-1, 18744).T)

preds_train = pd.read_csv(path_to_data + train_pred, index_col=0)
preds_train = pd.DataFrame(preds_train.values.reshape(-1, 59980).T)

preds_val = pd.read_csv(path_to_data + val_pred, index_col=0)
preds_val = pd.DataFrame(preds_val.values.reshape(-1, 14995).T)

for tgt in range(4):
    bins = np.linspace(-4, 4, 200)
    plt.figure(figsize=(12, 6))
    plt.hist(preds[tgt].values, bins=bins, alpha=.3, label="target "+str(tgt)+" test predictions")
    plt.hist(df_target_train[tgt].values, bins=bins, alpha=.3, label="target "+str(tgt)+" train")
    plt.hist(df_target_val[tgt].values, bins=bins, alpha=.3, label="target "+str(tgt)+" val")
    plt.legend()
    plt.title("Predictions for target "+str(tgt))
    plt.savefig('plot/target'+str(tgt)+'_train_val_pred.png')

    bins = np.linspace(-3, 3, 200)
    plt.figure(figsize=(12, 6))
    plt.hist((preds_train[tgt]-df_target_train[tgt]).values, bins=bins, alpha=.4, label="train error")
    plt.hist((preds_val[tgt]-df_target_val[tgt]).values, bins=bins, alpha=.4, label="val error")
    plt.legend()
    plt.title("Error for target "+str(tgt))
    plt.savefig('plot/target'+str(tgt)+'_error.png')
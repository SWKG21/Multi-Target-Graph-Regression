import sys


path_root = '..'
path_to_data = path_root + '/data/'
path_to_code = path_root + '/code/'
sys.path.insert(0, path_to_code)

all_preds = []
for tgt in range(4):
    with open(path_to_data + 'predictions_' + str(tgt) + '.txt', 'r') as file:
        all_preds.append(file.read().splitlines())

# flatten
all_preds = [pred for preds in all_preds for pred in pred]

with open(path_to_data + 'predictions_mchan.txt', 'w') as file:
    file.write('id,pred\n')
    for idx, pred in enumerate(all_preds):
        pred = format(pred, '.7f')
        file.write(str(idx) + ',' + pred + '\n')
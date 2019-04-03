import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

path_to_data = '../data/'


with open(path_to_data + 'targets/train/target_0.txt', 'r') as file:
    target0 = file.read().splitlines()
target0 = np.array([float(t) for t in target0])

with open(path_to_data + 'targets/train/target_1.txt', 'r') as file:
    target1 = file.read().splitlines()
target1 = np.array([float(t) for t in target1])

with open(path_to_data + 'targets/train/target_3.txt', 'r') as file:
    target3 = file.read().splitlines()
target3 = np.array([float(t) for t in target3])

tgt = 2

with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
    target = file.read().splitlines()
target = np.array([float(t) for t in target])

with open('predictions_'+ str(tgt) +'.txt') as file:
    preds = file.read().splitlines()
preds = np.array([float(pred) for pred in preds])


print (mean_squared_error(target, preds))

print ('post processing')
mses = []
min_error = mean_squared_error(target, preds)
best_c = 1.0
for c in np.arange(0.8, 1.2, 0.01):
    # new_error = mean_squared_error(target, np.log(np.exp(preds)*c))
    new_error = mean_squared_error(target, preds*c)
    if new_error < min_error:
        min_error = new_error
        best_c = c
    mses.append(new_error)

# print (best_c,min_error)
# plt.plot(np.arange(0.8, 1.2, 0.01), mses, '.')
# plt.show()

bins = np.linspace(-5, 5, 100)
plt.hist(preds, bins, alpha=0.3, label='pred 2')
plt.hist(target, bins, alpha=0.3, label='target 2')
# plt.hist(target0, bins, alpha=0.3, label='target 0')
# plt.hist(target1, bins, alpha=0.3, label='target 1')
# plt.hist(target3, bins, alpha=0.3, label='target 3')
plt.legend()
plt.show()



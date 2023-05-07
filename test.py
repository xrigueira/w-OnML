import numpy as np

y_preds = np.load('y_preds.npy', allow_pickle=True)

y_preds = [0 if (len(i)==0) or (i[0] >= 0.5) else 1 for i in y_preds]

print(y_preds[:5])

import numpy as np
import option
args=option.parse_args()

train_lines = [line.strip() for line in open(args.rgb_list)]
y_train = np.array([0 if 'Normal' in line else 1 for line in train_lines])

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, val_idx) in enumerate(skf.split(train_lines, y_train)):
    train_labels = y_train[train_idx]
    val_labels = y_train[val_idx]

    n_normal_train = np.sum(train_labels == 0)
    n_abnormal_train = np.sum(train_labels == 1)
    n_normal_val = np.sum(val_labels == 0)
    n_abnormal_val = np.sum(val_labels == 1)

    print(f"Fold {i+1}:")
    print(f"  Train -> Normal: {n_normal_train}, Abnormal: {n_abnormal_train}")
    print(f"  Val   -> Normal: {n_normal_val}, Abnormal: {n_abnormal_val}\n")

import option
args=option.parse_args()

import numpy as np
from sklearn.model_selection import StratifiedKFold

train_lines = [line.strip() for line in open(args.rgb_list)]

# labels: 0 = Normal, 1 = Abnormal
y_train = np.array([0 if 'Normal' in line else 1 for line in train_lines])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_files = []

for i, (train_idx, val_idx) in enumerate(skf.split(train_lines, y_train)):
    train_fold = [train_lines[j] for j in train_idx]
    val_fold = [train_lines[j] for j in val_idx]

    # Save for each fold
    with open(f'fold_{i+1}_train.txt', 'w') as f:
        f.write('\n'.join(train_fold))
    with open(f'fold_{i+1}_val.txt', 'w') as f:
        f.write('\n'.join(val_fold))

    fold_files.append((f'fold_{i+1}_train.txt', f'fold_{i+1}_val.txt'))

print("5-fold stratified train/val splits created!")
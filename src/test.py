import pandas as pd
from validation import hold_out
from validation import k_fold_cross_validation
from validation import leave_one_out

X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 1, 0, 1, 0]

# Aplicar Leave-One-Out
folds = leave_one_out(X, y)

for i, (X_train, X_val, y_train, y_val) in enumerate(folds, 1):
    print(f"Iteración {i}:")
    print(f"  X_train: {X_train}")
    print(f"  X_val: {X_val}")
    print(f"  y_train: {y_train}")
    print(f"  y_val: {y_val}")

import random

def hold_out(X, y, test_size=0.3):
    combined =  list(zip(X,y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test
    
def k_fold_cross_validation(X, y, k=10):
    combined = list(zip(X,y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    fold_size = len(X) // k
    folds = []
    for i in range(k):
        start = i*fold_size
        if i == k - 1:
            end = len(X)
        else:
            end = start + fold_size
        X_val = X[start:end]
        y_val = y[start:end]
        X_train = X[:start] + X[end:]
        y_train = y[:start] + y[end:]
        folds.append((X_train, X_val, y_train, y_val))
    return folds

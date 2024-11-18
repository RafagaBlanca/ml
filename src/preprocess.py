def min_max_scaler(X):
    X_T = list(zip(*X))
    scaled_X = []

    for feature in X_T:
        min_val = float(min(feature))
        max_val = float(max(feature))

        if max_val - min_val == 0:
            scaled_feature = [0.0] * len(feature)
        else:
            scaled_feature = [(x - min_val) / (max_val - min_val) for x in feature]

        scaled_X.append(scaled_feature)

    scaled_X = list(zip(*scaled_X))
    return [list(sample) for sample in scaled_X]

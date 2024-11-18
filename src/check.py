def accuracy_score(y_true, y_pred):
    correct = sum(1 for yt, yp in zip(y_true,y_pred) if yt == yp)
    return correct / len(y_true)

def recall_score_per_class(y_true, y_pred):
    from collections import Counter
    classes = set(y_true)
    recall_per_class = {}
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        recall_per_class[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall_per_class
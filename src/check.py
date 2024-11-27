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


def confusion_matrix_binary(y_true, y_pred):
    """
    Calcula la matriz de confusión para datasets binarios (clases 0 y 1).

    Args:
        y_true (list): Etiquetas verdaderas.
        y_pred (list): Predicciones.

    Returns:
        dict: Diccionario con TP, TN, FP, FN.
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(tp, fn):  # También llamado TPR
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def true_negative_rate(tn, fp):
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def false_positive_rate(fp, tn):
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

def false_negative_rate(fn, tp):
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0

def f1_score(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

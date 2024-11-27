'''
from check import *

# Ejemplo de datos
y_true = [0, 1, 1, 0, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

# Calcular Accuracy y Error
accuracy = accuracy_score(y_true, y_pred)
error = 1 - accuracy

# Generar Matriz de Confusión
conf_matrix = confusion_matrix_binary(y_true, y_pred)
tp, tn, fp, fn = conf_matrix["TP"], conf_matrix["TN"], conf_matrix["FP"], conf_matrix["FN"]

# Calcular Medidas
precision_value = precision(tp, fp)
recall_value = recall(tp, fn)
tnr = true_negative_rate(tn, fp)
fpr = false_positive_rate(fp, tn)
fnr = false_negative_rate(fn, tp)
f1 = f1_score(tp, fp, fn)

# Imprimir resultados
print(f"Accuracy: {accuracy:.4f}")
print(f"Error: {error:.4f}")
print(f"Precision: {precision_value:.4f}")
print(f"Recall: {recall_value:.4f}")
print(f"True Negative Rate: {tnr:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"False Negative Rate: {fnr:.4f}")
print(f"F1-Score: {f1:.4f}")
'''

from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
)

# Ejemplo de datos
y_true = [0, 1, 1, 0, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

# Accuracy y Matriz de Confusión
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# Precision, Recall, F1-Score
precision_value = precision_score(y_true, y_pred)
recall_value = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Imprimir resultados
print(f"Accuracy (sklearn): {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision (sklearn): {precision_value:.4f}")
print(f"Recall (sklearn): {recall_value:.4f}")
print(f"F1-Score (sklearn): {f1:.4f}")

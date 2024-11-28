import random
from collections import defaultdict

def hold_out(X, y, test_size=0.3, stratify=False, random_seed=None):
    """
    Divide un conjunto de datos en entrenamiento y prueba.

    Args:
        X (list): Características.
        y (list): Etiquetas.
        test_size (float): Proporción del conjunto de prueba.
        stratify (bool): Si True, realiza una división estratificada.
        random_seed (int): Semilla para la aleatorización.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if random_seed is not None:
        random.seed(random_seed)

    if stratify:
        # Crear un diccionario para agrupar por clase
        class_groups = defaultdict(list)
        for features, label in zip(X, y):
            class_groups[label].append(features)

        # Dividir cada clase en entrenamiento y prueba
        X_train, X_test, y_train, y_test = [], [], [], []
        for label, features in class_groups.items():
            test_count = int(len(features) * test_size)
            random.shuffle(features)
            X_test.extend(features[:test_count])
            y_test.extend([label] * test_count)
            X_train.extend(features[test_count:])
            y_train.extend([label] * (len(features) - test_count))
    else:
        # Mezclar aleatoriamente y dividir sin estratificación
        combined = list(zip(X, y))
        random.shuffle(combined)
        X, y = zip(*combined)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

    return list(X_train), list(X_test), list(y_train), list(y_test)


    
def k_fold_cross_validation(X, y, k=10, stratify=False, random_seed=None):
    """
    Divide los datos en K pliegues para validación cruzada, con opción de estratificación.

    Args:
        X (list): Datos de entrada.
        y (list): Etiquetas.
        k (int): Número de pliegues (folds).
        stratify (bool): Si True, realiza una división estratificada.
        random_seed (int): Semilla para la aleatorización.

    Returns:
        list: Lista de tuplas (X_train, X_val, y_train, y_val) para cada fold.
    """
    if k <= 1:
        raise ValueError("El número de pliegues (k) debe ser mayor que 1.")
    
    if random_seed is not None:
        random.seed(random_seed)
    
    if stratify:
        # Agrupar datos por clase
        class_groups = defaultdict(list)
        for features, label in zip(X, y):
            class_groups[label].append(features)
        
        # Dividir cada clase en K pliegues
        class_folds = {label: [] for label in class_groups}
        for label, samples in class_groups.items():
            random.shuffle(samples)
            fold_size = len(samples) // k
            for i in range(k):
                fold_samples = samples[i * fold_size:(i + 1) * fold_size]
                class_folds[label].append(fold_samples)
        
        # Crear pliegues combinados estratificados
        folds_data = []
        for i in range(k):
            X_val, y_val = [], []
            X_train, y_train = [], []
            
            for label, folds in class_folds.items():
                X_val.extend(folds[i])
                y_val.extend([label] * len(folds[i]))
                for j, fold in enumerate(folds):
                    if j != i:
                        X_train.extend(fold)
                        y_train.extend([label] * len(fold))
            
            folds_data.append((X_train, X_val, y_train, y_val))
    else:
        # Barajar los datos para aleatorizar los folds
        indices = list(range(len(X)))
        random.shuffle(indices)
        
        # Dividir los datos en k pliegues
        fold_size = len(X) // k
        folds_data = []
        for i in range(k):
            val_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = indices[:i * fold_size] + indices[(i + 1) * fold_size:]
            
            X_val = [X[idx] for idx in val_indices]
            y_val = [y[idx] for idx in val_indices]
            X_train = [X[idx] for idx in train_indices]
            y_train = [y[idx] for idx in train_indices]
            
            folds_data.append((X_train, X_val, y_train, y_val))
    
    return folds_data

def leave_one_out(X, y):
    """
    Divide los datos utilizando Leave-One-Out Cross-Validation.

    Args:
        X (list): Datos de entrada.
        y (list): Etiquetas.

    Returns:
        list: Lista de tuplas (X_train, X_val, y_train, y_val) para cada iteración.
    """
    folds_data = []
    n_samples = len(X)

    for i in range(n_samples):
        # Conjunto de validación: solo el elemento i
        X_val = [X[i]]
        y_val = [y[i]]
        
        # Conjunto de entrenamiento: todos menos el elemento i
        X_train = [X[j] for j in range(n_samples) if j != i]
        y_train = [y[j] for j in range(n_samples) if j != i]

        folds_data.append((X_train, X_val, y_train, y_val))

    return folds_data

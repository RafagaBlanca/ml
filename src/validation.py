import random

import random

def hold_out(X, y, test_size=0.3, random_seed=None):
    """
    Divide los datos en conjuntos de entrenamiento y prueba de acuerdo a un porcentaje definido por el usuario.

    Args:
        X (list): Datos de entrada.
        y (list): Etiquetas.
        test_size (float): Proporción de datos para el conjunto de prueba (0 < test_size < 1).
        random_seed (int, optional): Semilla para garantizar reproducibilidad.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if not (0 < test_size < 1):
        raise ValueError("El parámetro test_size debe estar entre 0 y 1.")

    if random_seed is not None:
        random.seed(random_seed)
    
    # Crear una lista de índices aleatorios
    indices = list(range(len(X)))
    random.shuffle(indices)

    # Calcular el tamaño del conjunto de prueba
    test_count = int(len(X) * test_size)

    # Dividir los datos en entrenamiento y prueba
    X_test = [X[i] for i in indices[:test_count]]
    y_test = [y[i] for i in indices[:test_count]]
    X_train = [X[i] for i in indices[test_count:]]
    y_train = [y[i] for i in indices[test_count:]]

    return X_train, X_test, y_train, y_test

    
def k_fold_cross_validation(X, y, k=10):
    """
    Divide los datos en K pliegues para validación cruzada.

    Args:
        X (list): Datos de entrada.
        y (list): Etiquetas.
        k (int): Número de pliegues (folds).

    Returns:
        list: Lista de tuplas (X_train, X_val, y_train, y_val) para cada fold.
    """
    if k <= 1:
        raise ValueError("El número de pliegues (k) debe ser mayor que 1.")

    # Barajar los datos para aleatorizar los folds
    indices = list(range(len(X)))
    random.shuffle(indices)

    # Dividir los datos en k pliegues
    fold_size = len(X) // k
    folds_data = []
    
    for i in range(k):
        # Índices del fold de validación
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        # Índices de entrenamiento (todos menos los de validación)
        train_indices = indices[:i * fold_size] + indices[(i + 1) * fold_size:]

        # Crear los conjuntos de entrenamiento y validación
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

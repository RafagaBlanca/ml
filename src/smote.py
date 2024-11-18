import random 
from collections import Counter
from knn import k_nn

def identify_minority_class(y):
    classes = Counter(y)
    min_class = min(classes, key=classes.get)
    return min_class, classes

def generate_samples(x, nn, N):
    samples = []
    for _ in range(N):
        #Seleccionar un vecino cualquiera
        x_j = random.choice(nn)
        #Generar un vector aleatorio 0 || 1
        gap = random.uniform(0,1)
        #Calculo de caracteristica generada
        sample = [xi + gap * (xj - xi) for xi, xj in zip(x,x_j)]

        samples.append(sample)

    return samples

def smote(X, y, N=100, k=10):
    
    #Identificar la clase minotaria y contar las muestras en X
    min_class, classes = identify_minority_class(y)
    min_class_idx = [i for i, label in enumerate(y) if label == min_class]
    min_samples = [X[i] for i in min_class_idx]

    #Muestar a generar por muestra original
    N = int(N/100)

    generated_samples = []
    for x_i in min_samples:
        nn = k_nn(x_i,min_samples,k)
        generated = generate_samples(x_i,nn,N)
        generated_samples.extend(generated)

    labels =  [min_class] * len(generated_samples)

    X = X + generated_samples
    y = y + labels

    return X,y

def smote_multiclass(X, y, k=5):
    class_counts = Counter(y)
    max_count = max(class_counts.values())  # Tamaño de la clase mayoritaria

    # Crear copias de X e y para agregar las muestras sintéticas
    X_resampled = X[:]
    y_resampled = y[:]

    for cls, count in class_counts.items():
        if count < max_count:
            # Identificar las muestras de la clase actual
            minority_indices = [i for i, label in enumerate(y) if label == cls]
            minority_samples = [X[i] for i in minority_indices]

            # Calcular cuántas muestras sintéticas necesitamos
            num_samples_to_generate = max_count - count

            # Generar muestras sintéticas para la clase actual
            synthetic_samples = []
            for x_i in minority_samples:
                neighbors = k_nn(x_i, minority_samples, k)
                synthetic_samples.extend(generate_samples(x_i, neighbors, num_samples_to_generate // len(minority_samples)))

            # Agregar las nuevas muestras al conjunto de datos
            X_resampled.extend(synthetic_samples[:num_samples_to_generate])
            y_resampled.extend([cls] * num_samples_to_generate)

    return X_resampled, y_resampled
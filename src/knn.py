class KNNClassifier:
    def __init__(self, k=1):
        """
        Inicializa el clasificador k-NN.

        Args:
            k (int): Número de vecinos más cercanos a considerar.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Almacena los datos de entrenamiento y etiquetas.

        Args:
            X (list): Datos de entrada de entrenamiento.
            y (list): Etiquetas correspondientes.
        """
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, a, b):
        """
        Calcula la distancia euclidiana entre dos puntos.

        Args:
            a (list): Punto 1.
            b (list): Punto 2.

        Returns:
            float: Distancia euclidiana.
        """
        return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)

    def predict(self, X_test):
        """
        Predice las clases para un conjunto de datos de prueba.

        Args:
            X_test (list): Datos de prueba.

        Returns:
            list: Predicciones de clase para cada dato de prueba.
        """
        predictions = []
        for x in X_test:
            # Calcular las distancias a todos los puntos de entrenamiento
            distances = [
                (self.euclidean_distance(x, x_train), y)
                for x_train, y in zip(self.X_train, self.y_train)
            ]

            # Ordenar por distancia y seleccionar los k vecinos más cercanos
            distances.sort(key=lambda x: x[0])
            k_nearest_neighbors = [label for _, label in distances[:self.k]]

            # Determinar la clase mayoritaria entre los vecinos más cercanos
            prediction = max(set(k_nearest_neighbors), key=k_nearest_neighbors.count)
            predictions.append(prediction)

        return predictions

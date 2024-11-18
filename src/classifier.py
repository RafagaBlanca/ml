def calculate_centroids(X_train, y_train):
    classes = list(set(y_train))
    centroids = {}
    for cls in classes:
        #Obtener todas las muestras de la clase actual
        class_samples = [X_train[i] for i in range(len(X_train)) if y_train[i] == cls]
        #Calcular el centroide de las muestras clase
        #Transpuesta de la matriz
        centroid = [sum(feature)/len(feature) for feature in zip(*class_samples)]
        centroids[cls] = centroid

    return centroids

def euclidean_distance(x1, x2):
    distance = 0.0
    for a, b in zip(x1,x2):
        distance += (a-b)**2
    return distance**0.5

class EuclideanClassifier:
    def __init__(self) -> None:
        self.centroids = {}
        pass

    def fit(self, X_train, y_train):
        if isinstance(y_train[0],list):
            y_train = [label[0] for label in y_train]
        #Calcular lso centroides de cada clase
        self.centroids = calculate_centroids(X_train,y_train)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            #Calcular la distancia al centroide de cada clase
            distances = {}
            for cls, centroid in self.centroids.items():
                dist = euclidean_distance(x,centroid)
                distances[cls] = dist
            #Asignar la claase con la distancia minima
            prediction = min(distances, key=distances.get)
            predictions.append(prediction)
        return predictions


class OneNN:
    def __init__(self) -> None:
        self.X_train = []
        self.y_train = []

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            min_distance = None
            min_class = None
            for x_train, y_train in zip(self.X_train, self.y_train):
                dist = euclidean_distance(x, x_train)
                if min_distance is None or dist < min_distance:
                    min_distance = dist
                    min_class = y_train
            predictions.append(min_class)
        return predictions

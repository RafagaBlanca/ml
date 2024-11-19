import random

class Perceptron:
    def __init__(self, n_features, learning_rate=0.1, max_epochs=100):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = [random.uniform(-0.1,0.1) for _ in range(n_features)]
        self.bias = random.uniform(-0.1,0.1)


    def activation_function(self, value):
        #Funcion del escalon unitario
        #Value es la sumatoria de las n_features + weights
        return 1 if value >= 0 else 0
        
    def predict(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.activation_function(weighted_sum)

    def fit(self, X, y):
        for epoch in range(self.max_epochs):
            errors = 0 
            for inputs, target in zip(X,y):
                prediction =  self.predict(inputs)
                error = target - prediction
                if error != 0:
                    #Update bias y weights
                    self.weights = [
                        w + (self.learning_rate * error * x) for w, x in zip(self.weights, inputs)
                    ]
                    self.bias = self.bias + (self.learning_rate * error)

            print(f"Epoch {epoch+1}: Errores: {errors}")


            if errors == 0:
                break

    def evaluate(self, X, y):
        correct = sum(1 for inputs, target in zip(X,y) if self.predict(inputs) == target)
        return correct

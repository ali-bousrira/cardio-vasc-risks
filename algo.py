# Importing libraries
import numpy as np

# Logistic Regression
class LogitRegression():
    def __init__(self, learning_rate=0.01, iterations=1000):        
        self.learning_rate = learning_rate        
        self.iterations = iterations
        self.W = None
        self.b = None
        self.cost_history = []
        
    def _sigmoid(self, z):
        """Numerically stable sigmoid function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
        
    def _cost_function(self, y_true, y_pred):
        """Calculate logistic regression cost (cross-entropy)"""
        m = len(y_true)
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
        
    # Function for model training    
    def fit(self, X, Y):        
        # no_of_training_examples, no_of_features        
        self.m, self.n = X.shape        
        
        # weight initialization        
        self.W = np.random.normal(0, 0.01, self.n)  # Small random initialization
        self.b = 0        
        self.X = X        
        self.Y = Y
        self.cost_history = []
        
        # gradient descent learning
        for i in range(self.iterations):            
            self.update_weights()
            
            # Calculate and store cost for monitoring
            if i % 100 == 0:  # Calculate cost every 100 iterations
                predictions = self._sigmoid(self.X.dot(self.W) + self.b)
                cost = self._cost_function(self.Y, predictions)
                self.cost_history.append(cost)
                
        return self
    
    # Helper function to update weights in gradient descent
    def update_weights(self):           
        # Forward propagation - using stable sigmoid
        z = self.X.dot(self.W) + self.b
        A = self._sigmoid(z)
        
        # Calculate gradients (vectorized)
        dW = (1/self.m) * self.X.T.dot(A - self.Y)
        db = (1/self.m) * np.sum(A - self.Y)
        
        # Update weights    
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db
        
        return self
    
    # Prediction function
    def predict(self, X):    
        z = X.dot(self.W) + self.b
        Z = self._sigmoid(z)
        Y = np.where(Z > 0.5, 1, 0)        
        return Y
    
    # Predict probabilities
    def predict_proba(self, X):
        z = X.dot(self.W) + self.b
        return self._sigmoid(z)
    
    # Calculate accuracy
    def score(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X_train, y_train):
        """Train the linear regression model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mse': mse,
            'r2': r2,
            'predictions': predictions
        } 
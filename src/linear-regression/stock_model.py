from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class StockPriceModel:
    def __init__(self, feature_days=60):
        """
        Initialize the stock price prediction model
        feature_days: number of previous days to use for prediction
        """
        self.model = LinearRegression()
        self.feature_days = feature_days
        
    def prepare_data(self, data):
        """
        Prepare the time series data for training
        Creates features from previous days' prices
        """
        X, y = [], []
        
        for i in range(self.feature_days, len(data)):
            X.append(data[i-self.feature_days:i])
            y.append(data[i])
            
        return np.array(X), np.array(y)
        
    def train(self, X_train, y_train):
        """Train the stock price prediction model"""
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
            'predictions': predictions,
            'actual': y_test
        } 
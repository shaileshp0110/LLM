from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class LogisticRegressionModel:
    def __init__(self):
        """Initialize the logistic regression model"""
        self.model = LogisticRegression(random_state=42)
        
    def train(self, X_train, y_train):
        """Train the logistic regression model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'predictions': predictions,
            'probabilities': probabilities
        } 
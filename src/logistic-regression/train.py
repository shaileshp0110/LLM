import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LogisticRegressionModel
from utils import plot_results

def generate_sample_data(n_samples=1000):
    """Generate sample data for binary classification"""
    np.random.seed(42)
    
    # Generate two features
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    
    # Generate target variable (binary classification)
    # If X1 + X2 > 0, class is 1, else 0
    y = (X1 + X2 + np.random.normal(0, 0.1, n_samples) > 0).astype(int)
    
    return pd.DataFrame({'feature1': X1, 'feature2': X2}), y

def main():
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train model
    print("Training logistic regression model...")
    model = LogisticRegressionModel()
    model.train(X_train_scaled, y_train)
    
    # Evaluate model
    results = model.evaluate(X_test_scaled, y_test)
    
    # Print results
    print("\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Plot results
    plot_results(X_test_scaled, y_test, results['predictions'], results['probabilities'])

if __name__ == "__main__":
    main() 
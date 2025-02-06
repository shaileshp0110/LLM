import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from simple_model import SimpleFraudDetectionModel
from utils import generate_sample_data, plot_results

def main():
    # Generate simple fraud detection data
    print("Generating simple fraud detection data...")
    X, y = generate_sample_data(n_samples=10000, fraud_ratio=0.10)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate
    model = SimpleFraudDetectionModel()
    model.train(X_train_scaled, y_train)
    results = model.evaluate(X_test_scaled, y_test)
    
    # Print results
    print("\nSimple Model Performance:")
    print(f"ROC AUC Score: {results['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot results
    plot_results(results['probabilities'], y_test)

if __name__ == "__main__":
    main() 
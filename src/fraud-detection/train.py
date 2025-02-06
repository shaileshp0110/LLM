import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import FraudDetectionModel
from utils import generate_realistic_fraud_data, plot_results

def main():
    # Generate realistic fraud detection data
    print("Generating realistic fraud detection data...")
    X, y, feature_names = generate_realistic_fraud_data(
        n_samples=10000,
        fraud_ratio=0.10,
        n_features=20
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train model
    print("\nTraining XGBoost fraud detection model...")
    model = FraudDetectionModel()
    model.train(X_train_scaled, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(X_test_scaled, y_test)
    
    # Print results
    print("\nModel Performance:")
    print(f"ROC AUC Score: {results['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Get and print feature importance
    importance = model.get_feature_importance(feature_names)
    print("\nTop 5 Most Important Features:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{feat}: {imp:.4f}")
    
    # Plot results
    plot_results(results['probabilities'], y_test)

if __name__ == "__main__":
    main() 
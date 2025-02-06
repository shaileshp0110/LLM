import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from realistic_model import RealisticFraudDetectionModel
from utils import generate_realistic_fraud_data, plot_results

def predict_transaction(model, scaler, transaction_data):
    """
    Predict if a single transaction is fraudulent
    
    Parameters:
    - transaction_data: dict with keys:
        - transaction_amount: float
        - time_of_day: float (0-24)
        - distance_from_last: float
        - transaction_frequency: int
    """
    # Convert transaction to model format
    features = np.array([[
        transaction_data['transaction_amount'],
        transaction_data['time_of_day'],
        transaction_data['distance_from_last'],
        transaction_data['transaction_frequency'],
        *[0] * 16  # Remaining features
    ]])
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Get probability of fraud
    fraud_probability = model.predict(scaled_features)[0]
    
    return {
        'is_fraud': fraud_probability > 0.5,
        'fraud_probability': fraud_probability,
        'risk_level': 'High' if fraud_probability > 0.8 else 
                     'Medium' if fraud_probability > 0.5 else 'Low'
    }

def main():
    # Generate realistic fraud detection data
    print("Generating realistic fraud detection data...")
    X, y, feature_names = generate_realistic_fraud_data(
        n_samples=10000,
        fraud_ratio=0.10
    )
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate
    model = RealisticFraudDetectionModel()
    model.train(X_train_scaled, y_train)
    results = model.evaluate(X_test_scaled, y_test)
    
    # Print results
    print("\nRealistic Model Performance:")
    print(f"ROC AUC Score: {results['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Get feature importance
    importance = model.get_feature_importance(feature_names)
    print("\nTop 5 Most Important Features:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{feat}: {imp:.4f}")
    
    # Plot results
    plot_results(results['probabilities'], y_test)

    # Example transactions with different risk levels
    example_transactions = [
        {
            'name': 'Normal Transaction',
            'transaction_amount': 50.0,     # Small amount
            'time_of_day': 14.0,           # 2 PM (normal hours)
            'distance_from_last': 5.0,      # Close to last transaction
            'transaction_frequency': 3       # Few recent transactions
        },
        {
            'name': 'Suspicious Transaction',
            'transaction_amount': 5000.0,    # Very large amount
            'time_of_day': 3.0,             # 3 AM (unusual hour)
            'distance_from_last': 1000.0,    # Very far from last transaction
            'transaction_frequency': 25      # Many transactions recently
        },
        {
            'name': 'Mixed Signals Transaction',
            'transaction_amount': 2000.0,    # Large amount
            'time_of_day': 13.0,            # 1 PM (normal hour)
            'distance_from_last': 300.0,     # Moderate distance
            'transaction_frequency': 15      # Moderate frequency
        }
    ]
    
    # Analyze each transaction
    print("\nTransaction Analysis:")
    print("-" * 60)
    for transaction in example_transactions:
        result = predict_transaction(model, scaler, transaction)
        print(f"\n{transaction['name']}:")
        print(f"Amount: ${transaction['transaction_amount']:.2f}")
        print(f"Time: {transaction['time_of_day']:02.0f}:00")
        print(f"Distance from last: {transaction['distance_from_last']} miles")
        print(f"Recent transactions: {transaction['transaction_frequency']}")
        print(f"Fraud Probability: {result['fraud_probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print("-" * 60)

if __name__ == "__main__":
    main() 
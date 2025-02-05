import pandas as pd
from sklearn.model_selection import train_test_split
from model import LinearRegressionModel
from utils import plot_results

def main():
    # Load data (example with random data)
    # Replace this with your actual data loading logic
    data = pd.DataFrame({
        'feature': range(100),
        'target': range(0, 200, 2)  # Example linear relationship
    })
    
    # Split features and target
    X = data[['feature']]
    y = data['target']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    model = LinearRegressionModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    results = model.evaluate(X_test, y_test)
    
    print(f"Model Performance:")
    print(f"MSE: {results['mse']:.4f}")
    print(f"R2 Score: {results['r2']:.4f}")
    
    # Plot results
    plot_results(X_test, y_test, results['predictions'])

if __name__ == "__main__":
    main() 
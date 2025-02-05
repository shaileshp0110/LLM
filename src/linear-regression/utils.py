import matplotlib.pyplot as plt
import pandas as pd

def plot_results(X_test, y_test, predictions):
    """
    Plot actual vs predicted values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, predictions, color='red', label='Predicted')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()

def plot_stock_predictions(symbol, actual, predictions, start_date):
    """
    Plot actual vs predicted stock prices
    """
    # Create date range for x-axis
    dates = pd.date_range(start=start_date, periods=len(actual), freq='B')
    
    plt.figure(figsize=(15, 7))
    plt.plot(dates, actual, color='blue', label='Actual Price', alpha=0.7)
    plt.plot(dates, predictions, color='red', label='Predicted Price', alpha=0.7)
    
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot prediction error
    plt.figure(figsize=(15, 7))
    plt.plot(dates, actual - predictions, color='green', label='Prediction Error')
    plt.title(f'{symbol} Prediction Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Error (Actual - Predicted)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show() 
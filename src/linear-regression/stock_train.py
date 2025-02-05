import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from stock_model import StockPriceModel
from utils import plot_stock_predictions
import numpy as np

def get_stock_data(symbol, start_date, end_date):
    """Download stock data using yfinance"""
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df['Close'].values

def train_stock_model(symbol="AAPL", start_date="2020-01-01", end_date="2023-12-31", feature_days=60):
    """
    Train and evaluate stock price prediction model
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'AAPL' for Apple)
    start_date : str
        Start date for historical data (YYYY-MM-DD)
    end_date : str
        End date for historical data (YYYY-MM-DD)
    feature_days : int
        Number of previous days to use for prediction
    """
    # Download stock data
    print(f"Downloading stock data for {symbol}...")
    stock_prices = get_stock_data(symbol, start_date, end_date)
    
    # Initialize model
    model = StockPriceModel(feature_days=feature_days)
    
    # Prepare data
    X, y = model.prepare_data(stock_prices)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Train model
    print("Training model...")
    model.train(X_train, y_train)
    
    # Evaluate model
    results = model.evaluate(X_test, y_test)
    
    print(f"\nModel Performance for {symbol}:")
    print(f"MSE: {results['mse']:.4f}")
    print(f"R2 Score: {results['r2']:.4f}")
    print(f"RMSE: {np.sqrt(results['mse']):.4f}")
    
    # Plot results
    plot_stock_predictions(
        symbol,
        results['actual'],
        results['predictions'],
        start_date=pd.Timestamp(start_date) + pd.Timedelta(days=feature_days)
    )
    
    return model, results

def main():
    # Example usage with different stocks
    stocks = [
        {
            'symbol': 'AAPL',
            'name': 'Apple'
        },
        {
            'symbol': 'GOOGL',
            'name': 'Google'
        },
        {
            'symbol': 'MSFT',
            'name': 'Microsoft'
        }
    ]
    
    for stock in stocks:
        print(f"\nAnalyzing {stock['name']} ({stock['symbol']})...")
        try:
            model, results = train_stock_model(
                symbol=stock['symbol'],
                start_date="2020-01-01",
                end_date="2023-12-31"
            )
            print(f"Analysis completed for {stock['symbol']}\n")
            print("-" * 50)
        except Exception as e:
            print(f"Error analyzing {stock['symbol']}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 
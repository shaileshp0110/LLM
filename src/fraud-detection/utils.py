import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def generate_sample_data(n_samples=10000, fraud_ratio=0.10, n_features=20):
    """
    Generate simple synthetic fraud detection data with clear separation
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Generate normal transactions (with simple pattern)
    normal_data = np.random.normal(0, 1, (n_normal, n_features))
    normal_labels = np.zeros(n_normal)
    
    # Generate fraudulent transactions (with different distribution)
    fraud_data = np.random.normal(2, 2, (n_fraud, n_features))
    fraud_labels = np.ones(n_fraud)
    
    # Combine the data
    X = np.vstack([normal_data, fraud_data])
    y = np.hstack([normal_labels, fraud_labels])
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

def generate_realistic_fraud_data(n_samples=10000, fraud_ratio=0.10, n_features=20):
    """
    Generate more realistic fraud detection data with overlapping distributions
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Initialize feature matrix
    X = np.zeros((n_samples, n_features))
    
    # Generate normal transactions
    normal_data = np.zeros((n_normal, n_features))
    
    # Transaction amount (higher for fraud)
    normal_data[:, 0] = np.random.lognormal(3, 1, n_normal)  # Most transactions $20-$100
    
    # Time of day (fraud more common at night)
    normal_data[:, 1] = np.random.normal(12, 4, n_normal)  # Center at noon
    normal_data[:, 1] = normal_data[:, 1] % 24  # Convert to 24-hour format
    
    # Distance from last transaction
    normal_data[:, 2] = np.random.exponential(10, n_normal)  # Most transactions nearby
    
    # Frequency of transactions
    normal_data[:, 3] = np.random.poisson(5, n_normal)  # Average 5 transactions
    
    # Random features with some correlation to fraud
    for i in range(4, n_features):
        normal_data[:, i] = np.random.normal(0, 1, n_normal)
    
    # Generate fraudulent transactions
    fraud_data = np.zeros((n_fraud, n_features))
    
    # Transaction amount (higher for fraud)
    fraud_data[:, 0] = np.random.lognormal(5, 1.5, n_fraud)  # Higher amounts
    
    # Time of day (fraud more common at night)
    fraud_data[:, 1] = np.random.normal(3, 2, n_fraud)  # Center at 3 AM
    fraud_data[:, 1] = fraud_data[:, 1] % 24
    
    # Distance from last transaction
    fraud_data[:, 2] = np.random.exponential(50, n_fraud)  # Larger distances
    
    # Frequency of transactions
    fraud_data[:, 3] = np.random.poisson(15, n_fraud)  # More frequent
    
    # Random features with some correlation to fraud
    for i in range(4, n_features):
        fraud_data[:, i] = np.random.normal(0.5, 1.2, n_fraud)
    
    # Add noise to make it more realistic
    normal_data += np.random.normal(0, 0.1, normal_data.shape)
    fraud_data += np.random.normal(0, 0.1, fraud_data.shape)
    
    # Combine the data
    X = np.vstack([normal_data, fraud_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    feature_names = [
        'transaction_amount',
        'time_of_day',
        'distance_from_last',
        'transaction_frequency'
    ] + [f'feature_{i}' for i in range(4, n_features)]
    
    return X, y, feature_names

def plot_results(y_prob, y_true):
    """
    Plot ROC curve and Precision-Recall curve with enhanced visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, 'b-', label=f'Model ROC (AUC = {auc(fpr, tpr):.3f})')
    ax1.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    
    # Add grid and labels
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ax2.plot(recall, precision, 'g-', label='Precision-Recall curve')
    ax2.plot([0, 1], [np.sum(y_true)/len(y_true)]*2, 'r--', 
             label='Random Classifier')
    
    # Add grid and labels
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot probability distribution
    plt.figure(figsize=(10, 5))
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.5, 
            label='Normal', density=True, color='blue')
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.5, 
            label='Fraud', density=True, color='red')
    plt.xlabel('Predicted Probability of Fraud')
    plt.ylabel('Density')
    plt.title('Probability Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show() 
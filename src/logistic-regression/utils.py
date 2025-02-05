import matplotlib.pyplot as plt
import numpy as np

def plot_results(X, y_true, y_pred, probabilities):
    """
    Plot the classification results
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Plot decision regions
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y_true, 
                         cmap=plt.cm.RdYlBu, alpha=0.6)
    ax1.set_title('Actual Classes')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax1)
    
    # Plot predictions
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=y_pred, 
                         cmap=plt.cm.RdYlBu, alpha=0.6)
    ax2.set_title('Predicted Classes')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    # Plot probability distributions
    plt.figure(figsize=(10, 6))
    for i in range(2):
        mask = y_true == i
        plt.hist(probabilities[mask][:, 1], bins=30, alpha=0.5, 
                label=f'Class {i}', density=True)
    plt.title('Probability Distributions by Class')
    plt.xlabel('Probability of Class 1')
    plt.ylabel('Density')
    plt.legend()
    plt.show() 
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_patient_data(n_samples=1000, n_features=6):
    """
    Generate synthetic patient data with natural clusters
    
    Features:
    - Age
    - BMI
    - Blood Pressure
    - Glucose Level
    - Cholesterol
    - Heart Rate
    """
    np.random.seed(42)
    
    # Define cluster centers and spreads
    centers = [
        # Young, healthy
        [25, 22, 110, 85, 170, 70],
        # Middle-aged, some health issues
        [45, 28, 130, 100, 220, 75],
        # Elderly, multiple conditions
        [70, 26, 145, 130, 240, 80]
    ]
    
    spreads = [
        [5, 2, 10, 10, 20, 5],    # Tight cluster
        [8, 3, 15, 15, 25, 8],    # Medium spread
        [10, 4, 20, 20, 30, 10]   # Wider spread
    ]
    
    # Generate samples for each cluster
    samples_per_cluster = n_samples // 3
    X = []
    
    for center, spread in zip(centers, spreads):
        cluster_samples = np.random.normal(
            loc=center,
            scale=spread,
            size=(samples_per_cluster, len(center))
        )
        X.append(cluster_samples)
    
    # Combine clusters and add some noise
    X = np.vstack(X)
    X += np.random.normal(0, 0.1, X.shape)
    
    # Ensure realistic ranges
    X[:, 0] = np.clip(X[:, 0], 18, 100)    # Age
    X[:, 1] = np.clip(X[:, 1], 15, 45)     # BMI
    X[:, 2] = np.clip(X[:, 2], 90, 180)    # Blood Pressure
    X[:, 3] = np.clip(X[:, 3], 70, 200)    # Glucose
    X[:, 4] = np.clip(X[:, 4], 150, 300)   # Cholesterol
    X[:, 5] = np.clip(X[:, 5], 60, 100)    # Heart Rate
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    
    feature_names = [
        'Age',
        'BMI',
        'Blood_Pressure',
        'Glucose_Level',
        'Cholesterol',
        'Heart_Rate'
    ]
    
    return X, feature_names

def preprocess_data(X):
    """Scale features to standard normal distribution"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler 
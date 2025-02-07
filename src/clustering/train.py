import numpy as np
from models import PatientClusteringModel
from utils import generate_patient_data, preprocess_data

def main():
    # Generate synthetic patient data
    print("Generating patient data...")
    X, feature_names = generate_patient_data(n_samples=1000)
    
    # Preprocess data
    print("Preprocessing data...")
    X_scaled, _ = preprocess_data(X)
    
    # Try different clustering algorithms
    algorithms = {
        'kmeans': {'n_clusters': 3},
        'dbscan': {'eps': 0.5, 'min_samples': 5},
        'hierarchical': {'n_clusters': 3}
    }
    
    for algo_name, params in algorithms.items():
        print(f"\nTraining {algo_name.upper()} clustering...")
        
        # Initialize and fit model
        model = PatientClusteringModel(algorithm=algo_name, params=params)
        model.fit(X_scaled)
        
        # Evaluate clustering
        results = model.evaluate(X_scaled)
        print("\nClustering Results:")
        print(f"Number of clusters: {results['n_clusters']}")
        print(f"Silhouette Score: {results['silhouette_score']:.3f}")
        print(f"Calinski-Harabasz Score: {results['calinski_harabasz_score']:.3f}")
        if results['n_noise'] > 0:
            print(f"Number of noise points: {results['n_noise']}")
        
        # Visualize results
        print("\nGenerating visualizations...")
        model.plot_clusters(X_scaled, feature_names)

if __name__ == "__main__":
    main() 
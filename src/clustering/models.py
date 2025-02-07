import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

class PatientClusteringModel:
    def __init__(self, algorithm='kmeans', params=None):
        """
        Initialize clustering model for patient segmentation
        
        Parameters:
        algorithm (str): 'kmeans', 'dbscan', or 'hierarchical'
        params (dict): Parameters for the chosen algorithm
        """
        self.algorithm = algorithm
        self.params = params or {}
        self.labels_ = None  # Store labels after fitting
        
        if algorithm == 'kmeans':
            self.model = KMeans(
                n_clusters=self.params.get('n_clusters', 3),
                random_state=42,
                **{k: v for k, v in self.params.items() if k != 'n_clusters'}
            )
        elif algorithm == 'dbscan':
            self.model = DBSCAN(
                eps=self.params.get('eps', 0.5),
                min_samples=self.params.get('min_samples', 5),
                **{k: v for k, v in self.params.items() if k not in ['eps', 'min_samples']}
            )
        elif algorithm == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.params.get('n_clusters', 3),
                **{k: v for k, v in self.params.items() if k != 'n_clusters'}
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def fit(self, X):
        """Fit the clustering model"""
        if self.algorithm == 'hierarchical':
            self.labels_ = self.model.fit_predict(X)
        else:
            self.model.fit(X)
            self.labels_ = self.model.labels_
        return self
    
    def predict(self, X):
        """Get cluster assignments"""
        if self.algorithm == 'hierarchical':
            # For hierarchical clustering, we can only return the labels from fit
            if X is self.last_X_fit:
                return self.labels_
            else:
                # Need to refit for new data
                return self.model.fit_predict(X)
        elif self.algorithm == 'kmeans':
            return self.model.predict(X)
        else:  # DBSCAN
            return self.model.fit_predict(X)
    
    def evaluate(self, X):
        """Evaluate clustering quality"""
        # For hierarchical clustering, use the labels from fit
        if self.algorithm == 'hierarchical':
            labels = self.labels_
        else:
            labels = self.predict(X)
        
        # Skip evaluation if all points are noise (-1) in DBSCAN
        if self.algorithm == 'dbscan' and all(l == -1 for l in labels):
            return {
                'silhouette_score': 0,
                'calinski_harabasz_score': 0,
                'n_clusters': 0,
                'n_noise': len(labels)
            }
        
        # Filter out noise points for DBSCAN
        if self.algorithm == 'dbscan':
            mask = labels != -1
            if sum(mask) < 2:  # Need at least 2 points for scoring
                return {
                    'silhouette_score': 0,
                    'calinski_harabasz_score': 0,
                    'n_clusters': 0,
                    'n_noise': sum(~mask)
                }
            X = X[mask]
            labels = labels[mask]
        
        results = {
            'silhouette_score': silhouette_score(X, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, labels),
            'n_clusters': len(np.unique(labels[labels != -1])),
            'n_noise': sum(labels == -1) if self.algorithm == 'dbscan' else 0
        }
        
        return results
    
    def plot_clusters(self, X, feature_names=None):
        """Visualize clusters using first two features or PCA"""
        # For hierarchical clustering, use the labels from fit
        if self.algorithm == 'hierarchical':
            labels = self.labels_
        else:
            labels = self.predict(X)
        
        # If more than 2 features, use only first two
        X_plot = X[:, :2] if X.shape[1] >= 2 else X
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        
        if feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            
        plt.title(f'Patient Clusters using {self.algorithm.upper()}')
        plt.show()
        
        # Plot feature distributions by cluster
        if feature_names:
            self._plot_feature_distributions(X, labels, feature_names)
    
    def _plot_feature_distributions(self, X, labels, feature_names):
        """Plot distribution of features for each cluster"""
        n_features = min(6, X.shape[1])  # Show up to 6 features
        n_rows = (n_features + 1) // 2
        
        plt.figure(figsize=(15, 4*n_rows))
        for i in range(n_features):
            plt.subplot(n_rows, 2, i+1)
            for cluster in np.unique(labels):
                if cluster == -1:  # Skip noise points in DBSCAN
                    continue
                cluster_data = X[labels == cluster, i]
                sns.kdeplot(data=cluster_data, label=f'Cluster {cluster}')
            plt.title(f'Distribution of {feature_names[i]}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
        
        plt.tight_layout()
        plt.show() 
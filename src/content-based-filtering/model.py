from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class ContentBasedRecommender:
    def __init__(self):
        """Initialize the content-based recommender system"""
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.cosine_sim = None
        self.movies = None
        self.indices = None
        
    def fit(self, movies_data):
        """
        Fit the recommender system with movie data
        
        Parameters:
        movies_data (pd.DataFrame): DataFrame containing movie information
            Required columns: 'title', 'genres', 'actors', 'description'
        """
        self.movies = movies_data
        
        # Combine features into a single text
        content_features = (
            movies_data['genres'].fillna('') + ' ' +
            movies_data['actors'].fillna('') + ' ' +
            movies_data['description'].fillna('')
        )
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf.fit_transform(content_features)
        
        # Calculate cosine similarity matrix
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create reverse mapping of movie titles and indices
        self.indices = pd.Series(movies_data.index, index=movies_data['title'])
        
    def get_recommendations(self, title, n_recommendations=5):
        """
        Get movie recommendations based on movie title
        
        Parameters:
        title (str): Title of the movie to base recommendations on
        n_recommendations (int): Number of recommendations to return
        
        Returns:
        list: List of dictionaries containing recommended movies and their similarity scores
        """
        # Get the index of the movie
        idx = self.indices[title]
        
        # Get similarity scores for all movies
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n_recommendations + 1]
        
        # Get movie indices and scores
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Return recommended movies with their details
        recommendations = []
        for idx, score in zip(movie_indices, similarity_scores):
            recommendations.append({
                'title': self.movies.iloc[idx]['title'],
                'genres': self.movies.iloc[idx]['genres'],
                'similarity_score': score
            })
            
        return recommendations 
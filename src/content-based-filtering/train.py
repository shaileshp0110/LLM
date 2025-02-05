import pandas as pd
from model import ContentBasedRecommender
from utils import load_sample_data, print_recommendations

def main():
    # Load sample movie data
    print("Loading movie data...")
    movies_df = load_sample_data()
    
    # Initialize and fit the recommender system
    print("Training content-based recommender system...")
    recommender = ContentBasedRecommender()
    recommender.fit(movies_df)
    
    # Get recommendations for a few example movies
    example_movies = [
        "The Dark Knight",
        "Toy Story",
        "The Matrix"
    ]
    
    for movie in example_movies:
        print(f"\nGetting recommendations for: {movie}")
        try:
            recommendations = recommender.get_recommendations(movie, n_recommendations=5)
            print_recommendations(recommendations)
        except KeyError:
            print(f"Movie '{movie}' not found in the database.")

if __name__ == "__main__":
    main() 
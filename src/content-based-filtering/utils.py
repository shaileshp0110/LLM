import pandas as pd
import numpy as np

def load_sample_data():
    """
    Create a sample movie dataset
    Returns a DataFrame with movie information
    """
    movies_data = {
        'title': [
            'The Dark Knight',
            'Inception',
            'The Matrix',
            'Toy Story',
            'Finding Nemo',
            'The Avengers',
            'Iron Man',
            'The Lion King',
            'Jurassic Park',
            'Avatar'
        ],
        'genres': [
            'Action, Crime, Drama',
            'Action, Adventure, Sci-Fi',
            'Action, Sci-Fi',
            'Animation, Adventure, Comedy',
            'Animation, Adventure, Comedy',
            'Action, Adventure, Sci-Fi',
            'Action, Adventure, Sci-Fi',
            'Animation, Adventure, Drama',
            'Action, Adventure, Sci-Fi',
            'Action, Adventure, Fantasy'
        ],
        'actors': [
            'Christian Bale, Heath Ledger',
            'Leonardo DiCaprio, Joseph Gordon-Levitt',
            'Keanu Reeves, Laurence Fishburne',
            'Tom Hanks, Tim Allen',
            'Albert Brooks, Ellen DeGeneres',
            'Robert Downey Jr., Chris Evans',
            'Robert Downey Jr., Gwyneth Paltrow',
            'Matthew Broderick, James Earl Jones',
            'Sam Neill, Laura Dern',
            'Sam Worthington, Zoe Saldana'
        ],
        'description': [
            'Batman fights against the Joker in Gotham City',
            'A thief enters dreams to plant ideas',
            'A computer programmer discovers a dystopian reality',
            'Toys come to life when humans are away',
            'A clownfish searches for his lost son',
            'Superheroes team up to save Earth',
            'Genius inventor builds powered armor suit',
            'Lion cub becomes king of the Pride Lands',
            'Dinosaurs are brought back to life',
            'Human becomes part of alien civilization'
        ]
    }
    
    return pd.DataFrame(movies_data)

def print_recommendations(recommendations):
    """
    Print movie recommendations in a formatted way
    """
    print("\nRecommended Movies:")
    print("-" * 60)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Genres: {rec['genres']}")
        print(f"   Similarity Score: {rec['similarity_score']:.4f}")
        print("-" * 60) 
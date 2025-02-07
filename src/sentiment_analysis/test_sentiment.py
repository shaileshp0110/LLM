import torch
from models import create_sentiment_analyzer
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

# Download all required NLTK data
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
except:
    print("Some NLTK downloads might have failed, but we can continue...")

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = word_tokenize(text.lower())
        
        # Convert tokens to indices
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(token_ids) < self.max_length:
            # Create mask (False for tokens, True for padding)
            mask = torch.zeros(self.max_length, dtype=torch.bool)
            mask[len(token_ids):] = True
            
            # Pad token_ids
            token_ids = token_ids + [self.vocab['<PAD>']] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
            mask = torch.zeros(self.max_length, dtype=torch.bool)
        
        return {
            'text': torch.tensor(token_ids),
            'mask': mask,
            'lengths': torch.tensor(min(len(tokens), self.max_length)),
            'labels': torch.tensor(self.labels[idx])
        }

def predict_sentiment(text, model, vocab, max_length=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Predict sentiment for a single text input"""
    model.eval()
    
    # Simple tokenization fallback if NLTK fails
    try:
        tokens = word_tokenize(text.lower())
    except:
        tokens = text.lower().split()
    
    # Convert tokens to indices
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Pad or truncate
    if len(token_ids) < max_length:
        mask = torch.zeros(max_length, dtype=torch.bool)
        mask[len(token_ids):] = True
        token_ids = token_ids + [vocab['<PAD>']] * (max_length - len(token_ids))
    else:
        token_ids = token_ids[:max_length]
        mask = torch.zeros(max_length, dtype=torch.bool)
    
    # Convert to tensors
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)  # Add batch dimension
    mask = mask.unsqueeze(0).to(device)
    length = torch.tensor([min(len(tokens), max_length)]).to(device)
    
    with torch.no_grad():
        if hasattr(model, 'lstm'):
            prediction = model(token_ids, length)
        else:
            prediction = model(token_ids, mask)
    
    probability = prediction.item()
    sentiment = "Positive" if probability >= 0.5 else "Negative"
    return sentiment, probability

# Define vocabulary at module level
vocab = {
    '<PAD>': 0,
    '<UNK>': 1,
    # Add common words
    'the': 2, 'a': 3, 'an': 4, 'and': 5, 'or': 6, 'but': 7,
    'movie': 8, 'film': 9, 'was': 10, 'is': 11, 'fantastic': 12,
    'terrible': 13, 'great': 14, 'awful': 15, 'loved': 16, 'hated': 17,
    'boring': 18, 'exciting': 19, 'brilliant': 20, 'worst': 21, 'best': 22,
    'masterpiece': 23, 'waste': 24, 'time': 25, 'good': 26, 'bad': 27,
    'okay': 28, 'special': 29, 'nothing': 30, 'very': 31, 'during': 32,
    'fell': 33, 'asleep': 34, 'predictable': 35, 'of': 36, 'it': 37,
    'i': 38, "i've": 39, 'seen': 40, 'complete': 41, 'every': 42,
    'minute': 43, 'modern': 44, 'cinema': 45, 'performances': 46
}

if __name__ == "__main__":
    # Example texts for testing
    example_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The worst film I've ever seen. Complete waste of time.",
        "It was okay, nothing special but not terrible either.",
        "A masterpiece of modern cinema, brilliant performances!",
        "I fell asleep during the movie, very boring and predictable."
    ]
    
    # Load the trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create and load the model (adjust parameters to match your trained model)
    model = create_sentiment_analyzer(
        model_type='transformer',  # or 'rnn'
        vocab_size=len(vocab),
        embedding_dim=256,
        n_heads=8,
        n_layers=3,
        dropout=0.5
    )
    
    # Load the trained weights if you have them
    try:
        model.load_state_dict(torch.load('best_model.pt', map_location=device))
        print("Loaded trained model successfully!")
    except:
        print("Could not load trained model, using untrained model for demonstration.")
    
    model = model.to(device)
    
    # Test the model on example texts
    print("\nTesting sentiment analysis:\n")
    for text in example_texts:
        sentiment, confidence = predict_sentiment(text, model, vocab)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 80) 
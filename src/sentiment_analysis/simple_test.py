import torch
from models import create_sentiment_analyzer

def simple_predict_sentiment(text, model, vocab, max_length=100):
    """Simplified prediction without NLTK dependency"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    # Simple tokenization
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
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(token_ids, mask)
    
    probability = prediction.item()
    sentiment = "Positive" if probability >= 0.5 else "Negative"
    return sentiment, probability

if __name__ == "__main__":
    # Simple vocabulary
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        'the': 2, 'a': 3, 'an': 4, 'and': 5, 'or': 6, 'but': 7,
        'movie': 8, 'film': 9, 'was': 10, 'is': 11, 'fantastic': 12,
        'terrible': 13, 'great': 14, 'awful': 15, 'loved': 16, 'hated': 17,
        'boring': 18, 'exciting': 19, 'brilliant': 20, 'worst': 21, 'best': 22
    }
    
    # Create and load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_sentiment_analyzer(
        model_type='transformer',
        vocab_size=len(vocab),
        embedding_dim=256,
        n_heads=8,
        n_layers=3,
        dropout=0.5
    ).to(device)
    
    # Try to load trained model
    try:
        model.load_state_dict(torch.load('best_model.pt', map_location=device))
        print("Loaded trained model successfully!")
    except:
        print("Using untrained model for demonstration.")
    
    # Test examples
    test_texts = [
        "This movie was fantastic!",
        "The worst film ever.",
        "It was okay, nothing special."
    ]
    
    print("\nTesting sentiment analysis:\n")
    for text in test_texts:
        sentiment, confidence = simple_predict_sentiment(text, model, vocab)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 80) 
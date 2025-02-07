import torch
from models import create_sentiment_analyzer

def initialize_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the saved model and vocabulary
    checkpoint = torch.load('best_model.pt', map_location=device)
    vocab = checkpoint['vocab']
    
    model = create_sentiment_analyzer(
        model_type='transformer',
        vocab_size=len(vocab),
        embedding_dim=256,
        n_heads=8,
        n_layers=3,
        dropout=0.1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded trained model with validation accuracy: {checkpoint['val_accuracy']:.4f}")
    
    return model.to(device), vocab

def predict_sentiment(text, model, vocab, max_length=256):
    device = next(model.parameters()).device
    model.eval()
    
    # Tokenize and convert to indices
    tokens = text.lower().split()
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

def main():
    model, vocab = initialize_model()
    
    print("\nWelcome to the IMDB Sentiment Analyzer!")
    print("Enter 'quit' to exit")
    print("-" * 50)
    
    while True:
        text = input("\nEnter your movie review: ")
        if text.lower() == 'quit':
            break
        
        sentiment, confidence = predict_sentiment(text, model, vocab)
        print(f"\nSentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 
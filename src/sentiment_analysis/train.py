import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

# Create a simple dummy dataset
class DummySentimentDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=50):
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        # Create dummy data
        self.texts = torch.randint(0, 1000, (num_samples, seq_length))  # Random token ids
        self.labels = torch.randint(0, 2, (num_samples,))  # Random binary labels
        self.lengths = torch.randint(10, seq_length, (num_samples,))  # Random sequence lengths
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        length = self.lengths[idx]
        # Create proper attention mask (1 for padding, 0 for actual tokens)
        mask = torch.zeros(self.seq_length, dtype=torch.bool)
        mask[length:] = True  # Mark padding positions as True
        
        return {
            'text': self.texts[idx],
            'lengths': self.lengths[idx],
            'mask': mask,
            'labels': self.labels[idx]
        }

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                n_epochs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Training loop for sentiment analysis models
    
    Args:
        model: The sentiment analyzer model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        n_epochs: Number of training epochs
        device: Device to train on
    """
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            optimizer.zero_grad()
            
            # For RNN
            if hasattr(model, 'lstm'):
                text, text_lengths = batch['text'], batch['lengths']
                text = text.to(device)
                predictions = model(text, text_lengths)
            # For Transformer
            else:
                text, mask = batch['text'], batch['mask']
                text = text.to(device)
                mask = mask.to(device) if mask is not None else None
                predictions = model(text, mask)
            
            labels = batch['labels'].to(device)
            loss = criterion(predictions.squeeze(), labels.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Similar logic as training
                if hasattr(model, 'lstm'):
                    text, text_lengths = batch['text'], batch['lengths']
                    text = text.to(device)
                    predictions = model(text, text_lengths)
                else:
                    text, mask = batch['text'], batch['mask']
                    text = text.to(device)
                    mask = mask.to(device) if mask is not None else None
                    predictions = model(text, mask)
                
                labels = batch['labels'].to(device)
                loss = criterion(predictions.squeeze(), labels.float())
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted_labels = (predictions.squeeze() > 0.5).long()
                correct_preds += (predicted_labels == labels).sum().item()
                total_preds += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_preds / total_preds
        
        print(f'Epoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        print(f'Validation accuracy: {accuracy:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print('Saved best model!')
        
        print('-' * 60)

# Main execution
if __name__ == "__main__":
    from models import create_sentiment_analyzer
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create dummy datasets
    train_dataset = DummySentimentDataset(num_samples=1000)
    val_dataset = DummySentimentDataset(num_samples=200)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model parameters
    vocab_size = 1000
    embedding_dim = 256
    
    # Create model (try both RNN and Transformer)
    print("Creating RNN model...")
    model = create_sentiment_analyzer(
        model_type='rnn',
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=128,
        n_layers=2,
        dropout=0.5
    )
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting training...")
    print("Device:", 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=5  # Reduced epochs for demonstration
    )
    
    print("\nTraining completed!")
    
    # Try Transformer model
    print("\nCreating Transformer model...")
    model = create_sentiment_analyzer(
        model_type='transformer',
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_heads=8,
        n_layers=3,
        dropout=0.5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting training with Transformer...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=5  # Reduced epochs for demonstration
    )
    
    print("\nTransformer training completed!") 
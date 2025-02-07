import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from collections import Counter

class IMDBDatasetLoader(Dataset):
    def __init__(self, split='train', max_length=256):
        # Load IMDB dataset from HuggingFace
        print(f"Loading {split} dataset...")
        dataset = load_dataset('imdb', split=split)
        self.texts = dataset['text']
        self.labels = dataset['label']
        self.max_length = max_length
        
        # Build vocabulary from training data
        if split == 'train':
            print("Building vocabulary...")
            word_freq = Counter()
            for text in tqdm(self.texts):
                words = text.lower().split()
                word_freq.update(words)
            
            # Create vocabulary (keep most common words)
            self.vocab = {
                '<PAD>': 0,
                '<UNK>': 1,
            }
            
            # Add most common words
            for word, freq in word_freq.most_common(9998):  # Keep top 10000 words
                self.vocab[word] = len(self.vocab)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx].lower()
        tokens = text.split()
        
        # Convert tokens to indices
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(token_ids) < self.max_length:
            mask = torch.zeros(self.max_length, dtype=torch.bool)
            mask[len(token_ids):] = True
            token_ids = token_ids + [self.vocab['<PAD>']] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
            mask = torch.zeros(self.max_length, dtype=torch.bool)
        
        return {
            'text': torch.tensor(token_ids),
            'mask': mask,
            'lengths': torch.tensor(min(len(tokens), self.max_length)),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                n_epochs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print("Training...")
        
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            text = batch['text'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(text, mask)
            loss = criterion(predictions.squeeze(), labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted_labels = (predictions.squeeze() > 0.5).float()
            correct_preds += (predicted_labels == labels).sum().item()
            total_preds += labels.size(0)
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds
        
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct_preds = 0
        total_preds = 0
        
        print("\nValidating...")
        with torch.no_grad():
            for batch in tqdm(val_loader):
                text = batch['text'].to(device)
                mask = batch['mask'].to(device)
                labels = batch['labels'].to(device)
                
                predictions = model(text, mask)
                loss = criterion(predictions.squeeze(), labels)
                val_loss += loss.item()
                
                predicted_labels = (predictions.squeeze() > 0.5).float()
                correct_preds += (predicted_labels == labels).sum().item()
                total_preds += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': train_dataset.vocab,
                'val_accuracy': val_accuracy
            }, 'best_model.pt')
            print('Saved best model!')

if __name__ == "__main__":
    print("Loading IMDB dataset...")
    
    # Create datasets
    train_dataset = IMDBDatasetLoader(split='train')
    test_dataset = IMDBDatasetLoader(split='test')
    test_dataset.vocab = train_dataset.vocab  # Share vocabulary
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    from models import create_sentiment_analyzer
    
    model = create_sentiment_analyzer(
        model_type='transformer',
        vocab_size=len(train_dataset.vocab),
        embedding_dim=256,
        n_heads=8,
        n_layers=3,
        dropout=0.1
    )
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    print("\nStarting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=3  # Increase for better results
    )
    
    print("\nTraining completed!") 
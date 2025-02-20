import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import VanishingGradientNet, ReLUSolution, BatchNormSolution, ResNetSolution
import matplotlib.pyplot as plt
from utils import plot_gradient_flow, plot_training_comparison

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    losses = []
    gradient_norms = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_grad_norm = 0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Calculate gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_grad_norm += total_norm
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_grad_norm = epoch_grad_norm / num_batches
        losses.append(avg_loss)
        gradient_norms.append(avg_grad_norm)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Average Gradient Norm: {avg_grad_norm:.4f}')
    
    return losses, gradient_norms

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    
    # Train different models
    models = {
        'Vanilla (Sigmoid)': VanishingGradientNet(),
        'ReLU': ReLUSolution(),
        'BatchNorm': BatchNormSolution(),
        'ResNet': ResNetSolution()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f'\nTraining {name} model...')
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        losses, grad_norms = train_model(
            model, train_loader, criterion, optimizer, device, epochs
        )
        
        results[name] = {
            'losses': losses,
            'gradient_norms': grad_norms
        }
    
    # Plot results
    plot_training_comparison(results)
    plot_gradient_flow(results)

if __name__ == '__main__':
    main() 
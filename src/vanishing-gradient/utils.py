import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_gradient_flow(results):
    """Plot gradient norms over time for different models"""
    plt.figure(figsize=(12, 6))
    
    for name, data in results.items():
        plt.plot(data['gradient_norms'], label=name)
    
    plt.title('Gradient Flow Analysis')
    plt.xlabel('Epoch')
    plt.ylabel('Average Gradient Norm')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_comparison(results):
    """Plot training losses for different models"""
    plt.figure(figsize=(12, 6))
    
    for name, data in results.items():
        plt.plot(data['losses'], label=name)
    
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_activations(model, loader, device):
    """Analyze activation distributions in different layers"""
    activations = []
    
    def hook(module, input, output):
        activations.append(output.detach().cpu())
    
    # Register hooks
    handles = []
    for layer in model.layers:
        handles.append(layer.register_forward_hook(hook))
    
    # Get activations
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            _ = model(data)
            break
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Plot activation distributions
    plt.figure(figsize=(15, 5))
    for i, act in enumerate(activations):
        plt.subplot(1, len(activations), i+1)
        plt.hist(act.numpy().flatten(), bins=50, density=True)
        plt.title(f'Layer {i+1}')
        plt.xlabel('Activation Value')
    plt.tight_layout()
    plt.show() 
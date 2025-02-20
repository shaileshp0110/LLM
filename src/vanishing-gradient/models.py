import torch
import torch.nn as nn

class VanishingGradientNet(nn.Module):
    """Network demonstrating vanishing gradients"""
    def __init__(self, input_size=784, hidden_size=256, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))  # Sigmoid causes vanishing gradients
        return self.output(x)

class ReLUSolution(nn.Module):
    """Solution using ReLU activation"""
    def __init__(self, input_size=784, hidden_size=256, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))  # ReLU helps with vanishing gradients
        return self.output(x)

class BatchNormSolution(nn.Module):
    """Solution using Batch Normalization"""
    def __init__(self, input_size=784, hidden_size=256, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.extend([
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            ])
        self.output = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

class ResNetSolution(nn.Module):
    """Solution using residual connections"""
    def __init__(self, input_size=784, hidden_size=256, num_layers=10):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size)
            ) for _ in range(num_layers // 2)
        ])
        
        self.output = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.res_blocks:
            identity = x
            x = block(x)
            x = torch.relu(x + identity)  # Residual connection
            
        return self.output(x) 
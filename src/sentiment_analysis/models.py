import torch
import torch.nn as nn

class RNNSentimentAnalyzer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Number of hidden units in LSTM
            n_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(RNNSentimentAnalyzer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           n_layers,
                           bidirectional=True,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        # Multiply hidden_dim by 2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        # text shape: (batch_size, seq_length)
        
        embedded = self.dropout(self.embedding(text))
        # Pack padded sequence for LSTM efficiency
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, 
                                                          batch_first=True, 
                                                          enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Concat the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        output = self.dropout(hidden)
        prediction = self.sigmoid(self.fc(output))
        
        return prediction

class TransformerSentimentAnalyzer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads, n_layers, dropout=0.5):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super(TransformerSentimentAnalyzer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text, mask=None):
        # text shape: (batch_size, seq_length)
        
        embedded = self.embedding(text)
        embedded = self.pos_encoder(embedded)
        
        # Apply transformer encoder
        # Note: PyTorch transformer uses mask=True for positions to be masked
        if mask is not None:
            output = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        else:
            output = self.transformer_encoder(embedded)
            
        # Use masked mean pooling
        if mask is not None:
            # Create mask for averaging (1 for valid tokens, 0 for padding)
            mask_expanded = (~mask).float().unsqueeze(-1)
            # Mask the output and compute mean
            masked_output = output * mask_expanded
            # Sum and divide by number of valid tokens
            pooled = masked_output.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = output.mean(dim=1)
        
        pooled = self.dropout(pooled)
        prediction = self.sigmoid(self.fc(pooled))
        
        return prediction

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Example usage:
def create_sentiment_analyzer(model_type='rnn', **kwargs):
    """
    Factory function to create sentiment analyzer models
    
    Args:
        model_type: 'rnn' or 'transformer'
        **kwargs: Model specific parameters
    """
    if model_type.lower() == 'rnn':
        return RNNSentimentAnalyzer(**kwargs)
    elif model_type.lower() == 'transformer':
        return TransformerSentimentAnalyzer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 
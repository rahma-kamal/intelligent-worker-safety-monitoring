import torch
import torch.nn as nn
from .base_har import BaseHARModel

class BiLSTMHARModel(BaseHARModel):
    """
    Bidirectional LSTM architecture for Temporal Human Activity Recognition.
    Processes sequences in both directions to capture complete temporal context.
    """
    def __init__(self, input_dim=92, num_classes=8, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Since it is bidirectional, hidden states are concatenated
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Inputs:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            out (torch.Tensor): Class logits of shape (batch_size, num_classes)
        """
        # Apply Layer Normalization to input features
        x = self.norm(x)
        
        # lstm_out shape: [batch_size, seq_len, hidden_size * 2]
        # h_n shape: [num_layers * 2, batch_size, hidden_size]
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling over time dimension to extract sequence features
        # pooling shape: [batch_size, hidden_size * 2]
        pooled_out = torch.mean(lstm_out, dim=1)
        
        logits = self.fc(pooled_out)
        return logits
